import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from codes.types import Player


class Branch:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0


class TreeNode:
    def __init__(self, state, value, priors, parent, last_move_idx):
        self.state = state
        self.value = value
        self.parent = parent
        self.last_move_idx = last_move_idx
        self.total_visit_count = 1
        self.branches = {}
        for idx, p in priors.items():
            if state.check_valid_move_idx(idx):
                self.branches[idx] = Branch(p)
        self.children = {}

    def move_idxes(self):
        return self.branches.keys()

    def add_child(self, move_idx, child_node):
        self.children[move_idx] = child_node

    def has_child(self, move_idx):
        return move_idx in self.children

    def get_child(self, move_idx):
        return self.children[move_idx]

    def record_visit(self, move_idx, value):
        self.total_visit_count += 1
        self.branches[move_idx].visit_count += 1
        self.branches[move_idx].total_value += value

    def expected_value(self, move_idx):
        branch = self.branches[move_idx]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count

    def prior(self, move_idx):
        return self.branches[move_idx].prior

    def visit_count(self, move_idx):
        if move_idx in self.branches:
            return self.branches[move_idx].visit_count
        return 0


class ZeroAgent:
    def __init__(self, encoder, model, device, c=3.0, rounds_per_move=300, noise=True):
        self.encoder = encoder
        self.device = device
        self.c = c
        self.model = model.to(self.device)
        self.model.eval()
        self.noise = noise
        self.alpha = 0.3
        self.collector = None
        self.gui = None

        self.num_simulated_games = 0
        self.rounds_per_move = rounds_per_move
        self.epoch = 0

        self.lr = 1e-3
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def set_device(self, device):
        self.device = torch.device(device)
        self.model = self.model.to(self.device)

    def set_collector(self, collector):
        self.collector = collector

    def set_gui(self, gui):
        self.gui = gui

    def set_lr(self, lr):
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def set_agent_meta_data(self, rounds_per_move=None, epoch=None):
        if rounds_per_move is not None:
            self.rounds_per_move = rounds_per_move
        if epoch is not None:
            self.epoch = epoch

    def add_num_simulated_games(self, num):
        self.num_simulated_games += num

    def create_node(self, state, move_idx=None, parent=None):
        with torch.no_grad():
            state_tensor = self.encoder.encode(state)
            model_input = torch.tensor([state_tensor], dtype=torch.float, device=self.device)
            prior, value = self.model(model_input)
            prior = prior[0].detach()
            value = value[0][0].detach()

        if self.noise and parent is None:
            noise_probs = np.random.dirichlet(self.alpha * np.ones(self.encoder.num_moves()))
            move_priors = {
                idx: 0.75 * p + 0.25 * noise_probs[idx]
                for idx, p in enumerate(prior)
            }
        else:
            move_priors = {
                idx: p
                for idx, p in enumerate(prior)
            }

        new_node = TreeNode(state, value, move_priors, parent, move_idx)

        if parent is not None:
            parent.add_child(move_idx, new_node)

        return new_node

    def select_branch(self, node):
        total_n = node.total_visit_count

        def score_branch(move_idx):
            q = node.expected_value(move_idx)
            p = node.prior(move_idx)
            n = node.visit_count(move_idx)
            return q + self.c * p * np.sqrt(total_n) / (n + 1)

        return max(node.move_idxes(), key=score_branch)

    def select_move(self, game_state):
        """
        select move by mcts.
        return move idx
        """
        root = self.create_node(game_state)
        remain_rounds = self.rounds_per_move
        # 게임 진행
        for i in range(remain_rounds):
            node = root
            next_move_idx = self.select_branch(node)
            while node.has_child(next_move_idx):
                node = node.get_child(next_move_idx)
                if node.state.check_can_play() and (not node.state.check_game_over()):
                    next_move_idx = self.select_branch(node)

            # if node.state.game_over:
            #     move_idx = node.last_move_idx
            #     value = -1 * node.value
            #     node = node.parent
            # else:
            #     next_move = self.encoder.decode_move_index(next_move_idx)
            #     new_state = node.state.apply_move(next_move)
            #     child_node = self.create_node(new_state, next_move_idx, node)
            #     move_idx = next_move_idx
            #     value = -1 * child_node.value

            # 노드가 없거나 게임이 끝났으면 게임 진행 종료
            # 게임이 끝났으면
            if node.state.game_over:
                move_idx = node.last_move_idx
                # 승자가 없으면
                if node.state.winner == Player.both:
                    value = 0
                else:
                    value = -1  # parent의 player은 node의 player와 다름. node에서 승리했다면 parent는 패배
                node = node.parent
            else:
                next_move = self.encoder.decode_move_index(next_move_idx)
                new_state = node.state.apply_move(next_move)
                child_node = self.create_node(new_state, next_move_idx, node)
                move_idx = next_move_idx
                value = -1 * child_node.value

            while node is not None:
                node.record_visit(move_idx, value)
                move_idx = node.last_move_idx
                node = node.parent
                value = -1 * value

        visit_counts = np.array([
            root.visit_count(idx)
            for idx in range(self.encoder.num_moves())
        ])

        if self.collector is not None:
            root_state_tensor = self.encoder.encode(game_state)
            self.collector.record_decision(root_state_tensor, visit_counts)

        if self.gui is not None:
            self.gui.show_visit_counts(visit_counts)

        selected_move_idx = max(root.move_idxes(), key=root.visit_count)
        return self.encoder.decode_move_index(selected_move_idx)

    def train(self, experience, batch_size):
        self.model.train()

        dataloader = DataLoader(experience, batch_size=batch_size, num_workers=4,
                                shuffle=True, pin_memory=True)
        policy_loss_sum = 0
        value_loss_sum = 0
        epoch_loss_sum = 0
        epoch_samples = 0

        for state, p, reward in tqdm(dataloader):
            state = state.to(self.device).float()
            p = p.to(self.device).float()
            reward = reward.to(self.device).float()
            reward = reward.unsqueeze(1)
            epoch_samples += state.size(0)

            with torch.set_grad_enabled(True):
                output = self.model(state)
                # categorical crossentropy
                # output의 결과는 softmax
                policy_loss = -1 * torch.mean(torch.sum(p * output[0].log(), dim=-1))
                value_loss = F.mse_loss(output[1], reward)

                epoch_loss = policy_loss + value_loss

                policy_loss_sum += policy_loss.detach() * state.size(0)
                value_loss_sum += value_loss.detach() * state.size(0)
                epoch_loss_sum += epoch_loss.detach() * state.size(0)
                epoch_samples += state.size(0)

                self.optimizer.zero_grad()
                epoch_loss.backward()
                self.optimizer.step()

        policy_loss = policy_loss_sum / epoch_samples
        value_loss = value_loss_sum / epoch_samples
        loss = policy_loss + value_loss

        self.model.eval()

        return loss, policy_loss, value_loss

    def save_agent(self, pthfile):
        state = {
            'encoder': self.encoder,
            'model': self.model,
            'c': self.c,
            'rounds_per_move': self.rounds_per_move,
            'num_simulated_games': self.num_simulated_games,
            'epoch': self.epoch,
            'optimizer': self.optimizer.state_dict(),
        }

        torch.save(state, pthfile)

    @staticmethod
    def load_agent(pthfilename, device):
        loaded_file = torch.load(pthfilename, map_location='cpu')
        encoder = loaded_file['encoder']
        model = loaded_file['model']
        c = loaded_file['c']
        rounds_per_move = loaded_file['rounds_per_move']
        num_simulated_games = loaded_file['num_simulated_games']
        epoch = loaded_file['epoch']
        optimizer = loaded_file['optimizer']

        loaded_agent = ZeroAgent(encoder, model, device, c)
        loaded_agent.optimizer.load_state_dict(optimizer)
        loaded_agent.set_agent_meta_data(rounds_per_move=rounds_per_move, epoch=epoch)
        print("simulated %d games." % num_simulated_games)
        return loaded_agent
