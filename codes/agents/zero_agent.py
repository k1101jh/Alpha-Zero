# 참고한 코드: https://github.com/maxpumperla/deep_learning_and_the_game_of_go/blob/master/code/dlgo/zero/agent.py
import numpy as np
import heapq
import threading
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from codes.agents.abstract_agent import Agent
from codes.game_types import Player


class Branch:
    def __init__(self, prior):
        """[summary]

        Args:
            prior (float): Prior of this branch.
        """

        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0


class TreeNode:
    def __init__(self, state, value, priors, parent, last_move_idx):
        """[summary]
            MCTS tree node.
            Each node has game state.
        Args:
            state (GameState): Game state.
            value (float): Value of this state.
            priors (dict): Prior of branches. Key is index and value is prior.
            parent (TreeNode): Parent node.
            last_move_idx (int): Last move index.
        """

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


class ZeroAgent(Agent):
    def __init__(self, encoder, model, device, c=5.0, rounds_per_move=300, num_threads_per_round=12, noise=True, lr=1e-3):
        """[summary]
            Use Monte Carlo Tree Search algorithm with DeepLearning.
        Args:
            encoder ([type]): [description]
            model ([type]): [description]
            device ([type]): [description]
            c (float, optional): [description]. Defaults to 5.0.
            rounds_per_move (int, optional): [description]. Defaults to 300.
            num_threads_per_round (int, optional): [description]. Defaults to 12.
            noise (bool, optional): [description]. Defaults to True.
            lr (float, optional): [description]. Defaults to 1e-3.
        """

        super().__init__()
        self.encoder = encoder
        self.device = device
        self.c = c
        self.model = model.to(self.device)
        self.model.eval()
        self.noise = noise
        self.alpha = 0.2
        self.collector = None

        self.num_simulated_games = 0
        self.num_threads_per_round = num_threads_per_round
        self.rounds_per_move = rounds_per_move
        self.epoch = 0

        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 250, 0.1)

    def __deepcopy__(self, memodict={}):
        """[summary]

        Args:
            memodict (dict, optional): [description]. Defaults to {}.

        Returns:
            ZeroAgent: Copied ZeroAgent.
        """

        copy_object = ZeroAgent(self.encoder, self.model, self.device, self.c, self.rounds_per_move, self.num_threads_per_round, self.noise)
        copy_object.set_collector(self.collector)
        copy_object.num_simulated_games = self.num_simulated_games
        copy_object.epoch = self.epoch
        copy_object.optimizer.load_state_dict(self.optimizer.state_dict())
        copy_object.scheduler.load_state_dict(self.scheduler.state_dict())

        return copy_object

    def set_noise(self, noise):
        self.noise = noise

    def set_collector(self, collector):
        self.collector = collector

    def set_agent_data(self, epoch=None, num_simulated_games=None):
        if epoch is not None:
            self.epoch = epoch

        if num_simulated_games is not None:
            self.num_simulated_games = num_simulated_games

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

    def select_branches(self, node, num):
        total_n = node.total_visit_count

        def score_branch(move_idx):
            q = node.expected_value(move_idx)
            p = node.prior(move_idx)
            n = node.visit_count(move_idx)
            return q + self.c * p * np.sqrt(total_n) / (n + 1)

        return heapq.nlargest(num, node.move_idxes(), key=score_branch)

    def select_move(self, game_state):
        root = self.create_node(game_state)
        remain_rounds = self.rounds_per_move

        # 게임 진행
        for _ in tqdm(range(remain_rounds)):
            first_move_idx_candidates = self.select_branches(root, self.num_threads_per_round)
            threads = []
            thread_lock = threading.Lock()
            for first_move_idx_candidate in first_move_idx_candidates:
                t = threading.Thread(target=self.simulate, args=(root, first_move_idx_candidate, thread_lock))
                t.daemon = True
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

        visit_counts = np.array([
            root.visit_count(idx)
            for idx in range(self.encoder.num_moves())
        ])

        if self.collector is not None:
            root_state_tensor = self.encoder.encode(game_state)
            self.collector.record_decision(root_state_tensor, visit_counts)

        selected_move_idx = max(root.move_idxes(), key=root.visit_count)
        return self.encoder.decode_move_index(selected_move_idx), visit_counts

    def simulate(self, root, next_move_idx, thread_lock):
        node = root
        while node.has_child(next_move_idx):
            node = node.get_child(next_move_idx)
            if not node.state.game_over:
                next_move_idx = self.select_branch(node)

        # 노드가 없거나 승자가 나왔으면 게임 진행 종료
        # 게임이 끝났으면
        if node.state.game_over:
            move_idx = node.last_move_idx
            # 승자가 없으면(돌을 더 놓을 수 없으면)
            if node.state.winner == Player.both:
                value = 0
            else:
                value = 1

            node = node.parent
        else:
            next_move = self.encoder.decode_move_index(next_move_idx)
            new_state = node.state.apply_move(next_move)
            child_node = self.create_node(new_state, next_move_idx, parent=node)

            move_idx = next_move_idx
            if child_node.state.check_game_over():
                if child_node.state.winner == Player.both:
                    value = 0
                else:
                    value = 1
            else:
                value = -1 * child_node.value.item()

        thread_lock.acquire()
        while node is not None:
            node.record_visit(move_idx, value)
            move_idx = node.last_move_idx
            node = node.parent
            value = -1 * value
        thread_lock.release()

    def train(self, dataset, batch_size):
        self.model.train()

        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4,
                                shuffle=False, pin_memory=True)
        policy_loss_sum = 0
        value_loss_sum = 0
        epoch_loss_sum = 0
        dataset_size = len(dataset)

        for state, p, reward in tqdm(dataloader):
            state = state.to(self.device).float()
            p = p.to(self.device).float()
            reward = reward.to(self.device).float()
            reward = reward.unsqueeze(1)

            with torch.set_grad_enabled(True):
                output = self.model(state)
                # categorical crossentropy
                # output의 결과는 softmax
                policy_loss = -1 * torch.mean(torch.sum(p * output[0].log(), dim=1))
                value_loss = F.mse_loss(output[1], reward)

                epoch_loss = policy_loss + value_loss

                policy_loss_sum += policy_loss.detach() * state.size(0)
                value_loss_sum += value_loss.detach() * state.size(0)
                epoch_loss_sum += epoch_loss.detach() * state.size(0)

                self.optimizer.zero_grad()
                epoch_loss.backward()
                self.optimizer.step()

        self.scheduler.step()

        policy_loss = policy_loss_sum / dataset_size
        value_loss = value_loss_sum / dataset_size
        loss = policy_loss + value_loss

        self.model.eval()

        return loss, policy_loss, value_loss

    def save_agent(self, pthfile):
        state = {
            'encoder': self.encoder,
            'model': self.model,
            'rounds_per_move': self.rounds_per_move,
            'num_simulated_games': self.num_simulated_games,
            'epoch': self.epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }

        torch.save(state, pthfile)

    @staticmethod
    def load_agent(pthfilename, device, num_threads_per_round, noise=False):
        loaded_file = torch.load(pthfilename, map_location='cuda:0')
        encoder = loaded_file['encoder']
        model = loaded_file['model']
        rounds_per_move = loaded_file['rounds_per_move']
        num_simulated_games = loaded_file['num_simulated_games']
        epoch = loaded_file['epoch']
        optimizer_state_dict = loaded_file['optimizer']
        scheduler_state_dict = loaded_file['scheduler']

        loaded_agent = ZeroAgent(encoder, model, device,
                                 rounds_per_move=rounds_per_move,
                                 num_threads_per_round=num_threads_per_round,
                                 noise=noise)
        loaded_agent.optimizer.load_state_dict(optimizer_state_dict)
        loaded_agent.scheduler.load_state_dict(scheduler_state_dict)
        loaded_agent.set_agent_data(epoch=epoch, num_simulated_games=num_simulated_games)
        print("simulated %d games." % num_simulated_games)
        return loaded_agent
