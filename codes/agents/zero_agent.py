import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from codes.types import Player


class TreeNode:
    def __init__(self, parent, prior, c):
        self.parent = parent
        self.children = {}
        self.c = c
        self.value = 0

        self.q = 0
        self.p = prior  # prior
        self.n = 0      # visit count

    def add_child(self, move_idx, child_node):
        self.children[move_idx] = child_node

    def record_visit(self, value):
        self.n += 1

    def is_leaf(self):
        return self.children == {}


class ZeroAgent:
    def __init__(self, encoder, model, device, c=3.0, noise=True):
        self.encoder = encoder
        self.device = device
        self.c = c
        self.model = model.to(self.device)
        self.model.eval()
        self.noise = noise
        self.alpha = 0.3

        self.rounds_per_move = 300
        self.epoch = 0

    def set_data(self, rounds_per_move=None, epoch=None):
        if rounds_per_move is not None:
            self.rounds_per_move = rounds_per_move
        if epoch is not None:
            self.epoch = epoch

    def create_node(self, state, move_idx, parent=None):
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

        new_node = TreeNode(parent, move_priors, self.c)

        if parent is not None:
            parent.add_child(move_idx, new_node)

        return new_node

    def select_move(self, game_state):
        """
        select move by mcts.
        return move idx
        """
        node = self.create_node(game_state)
        # 게임 진행
        while game_state.check_can_play():
            if node.is_leaf():
                break
            action, node = node.select()
            game_state = game_state.apply_move(action, change_turn=True)
            # 게임 종료 시
            if game_state.check_game_over():
                break

        value = node.value

        # 노드가 없거나 게임이 끝났으면 게임 진행 종료
        # 게임이 끝났으면
        if game_state.game_over:
            # 승자가 없으면
            if game_state.winner == Player.both:
                value = 0
            elif game_state.winner == game_state.player:
                value = 1
            # 상대가 이겼으면
            else:
                value = -1

        while node is not None:
            node.record_visit(value)
            node = node.parent
            value = -value

    def train(self, experience, batch_size):
        self.model.train()

        dataloader = DataLoader(experience, batch_size=batch_size, num_workers=4,
                                shuffle=True, pin_memory=True)
        policy_loss_sum = 0
        value_loss_sum = 0
        epoch_samples = 0

        for state, p, reward in tqdm(dataloader):
            state = state.to(self.device).float()
            p = p.to(self.device).float()
            reward = reward.to(self.device).float()
            epoch_samples += state.size(0)

            with torch.set_grad_enabled(True):
                output = self.model(state)

                policy_loss = -torch.mean(torch.sum(p * output[0], dim=1))
                value_loss = F.mse_loss(output[1], reward)

                loss = policy_loss + value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.scheduler.step()
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
            'rounds_per_move': self.num_rounds,
            'num_simulated_games': self.num_simulated_games,
            'epoch': self.epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
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
        scheduler = loaded_file['scheduler']

        loaded_agent = ZeroAgent(encoder, model, device, c)
        loaded_agent.optimizer.load_state_dict(optimizer)
        loaded_agent.scheduler.load_state_dict(scheduler)
        loaded_agent.set_data(rounds_per_move=rounds_per_move,
                              epoch=epoch)
        print("simulated %d games." % num_simulated_games)
        return loaded_agent
