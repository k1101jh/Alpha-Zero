# 참고한 코드: https://github.com/maxpumperla/deep_learning_and_the_game_of_go/blob/master/code/dlgo/zero/agent.py
from typing import Dict, Iterable, Optional, Tuple, TypeVar
import numpy as np
import heapq
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from agents.abstract_agent import AbstractAgent
from encoders.zero_encoder import ZeroEncoder
from games.experience import ExperienceCollector, ExperienceDataset
from games.game_components import Move, Player
from games.abstract_game_state import AbstractGameState


SelfTreeNode = TypeVar("SelfTreeNode", bound="TreeNode")
SelfZeroAgent = TypeVar("SelfZeroAgent", bound="ZeroAgent")


class TreeNode:
    def __init__(self, state: AbstractGameState, value: float, priors: Dict, parent: SelfTreeNode, c: float, last_move_idx: int):
        """[summary]
            MCTS tree node.
            Each node has game state.
        Args:
            state (AbstractGameState): Game state.
            value (float): Value of this state.
            priors (Dict): Prior of branches. Key is index and value is prior.
            parent (SelfTreeNode): Parent node.
            c (float): [description]
            last_move_idx (int): [description]
        """
        self.state = state
        self.value = value
        self.parent = parent
        self.c = c
        self.last_move_idx = last_move_idx
        self.total_visit_count = 1
        self.branches = {}
        num_points = self.state.get_board_size() ** 2
        self.q_scores = np.zeros((num_points), dtype=np.float64)
        self.c_scores = np.zeros((num_points), dtype=np.float64)
        for idx, p in priors.items():
            if state.check_valid_move_idx(idx):
                self.branches[idx] = {
                    "visit_count": 0,
                    "total_value": 0.0,
                    "p": p,
                }
                self.c_scores[idx] = self.c * p
        self.children = {}

    def move_idxes(self) -> Iterable[int]:
        return self.branches.keys()

    def add_child(self, move_idx: int, child_node: SelfTreeNode) -> None:
        self.children[move_idx] = child_node

    def has_child(self, move_idx: int) -> bool:
        return move_idx in self.children

    def get_child(self, move_idx: int) -> SelfTreeNode:
        return self.children[move_idx]

    def record_visit_and_update_score(self, move_idx: int, value: float) -> None:
        self.total_visit_count += 1
        self.branches[move_idx]["visit_count"] += 1
        self.branches[move_idx]["total_value"] += value

        visit_count = self.branches[move_idx]["visit_count"]
        p = self.branches[move_idx]["p"]
        self.q_scores[move_idx] = self.branches[move_idx]["total_value"] / visit_count
        self.c_scores[move_idx] = self.c * p / (visit_count + 1)

    def get_scores(self, sqrt_total_n: float) -> np.ndarray:
        return self.q_scores + self.c_scores * sqrt_total_n

    def visit_count(self, move_idx: int) -> int:
        if move_idx in self.branches:
            return self.branches[move_idx]["visit_count"]
        return 0


class ZeroAgent(AbstractAgent):
    def __init__(self, encoder: ZeroEncoder, model: nn.Module, device: str,
                 c: float = 5.0, simulations_per_move: int = 500, num_threads_per_round: int = 1,
                 noise: bool = True, lr: float = 1e-4):
        """[summary]
            Use Monte Carlo Tree Search algorithm with DeepLearning.
        Args:
            encoder ([type]): [description]
            model ([type]): [description]
            device ([type]): [description]
            c (float, optional): [description]. Defaults to 5.0.
            simulations_per_move (int, optional): [description]. Defaults to 300.
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
        self.simulations_per_move = simulations_per_move
        self.epoch = 0

        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=150, gamma=0.5)

    # def __deepcopy__(self, memo):
        # """[summary]
        # Args:
        #     memodict (dict, optional): [description]. Defaults to {}.

        # Returns:
        #     ZeroAgent: Copied ZeroAgent.
        # """
        # copy_object = ZeroAgent(self.encoder, self.model, self.device, self.c, self.simulations_per_move, self.num_threads_per_round, self.noise)
        # copy_object.set_collector(self.collector)
        # copy_object.num_simulated_games = self.num_simulated_games
        # copy_object.epoch = self.epoch
        # copy_object.optimizer.load_state_dict(self.optimizer.state_dict())
        # copy_object.scheduler.load_state_dict(self.scheduler.state_dict())

        # return copy_object
        
        # cls = self.__class__
        # result = cls.__new__(cls)
        # memo[id(self)] = result
        # for k, v in self.__dict__.items():
        #     setattr(result, k, deepcopy(v, memo))
        # return result

    def set_device(self, device: str) -> None:
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    def set_noise(self, noise: bool) -> None:
        self.noise = noise

    def set_collector(self, collector: ExperienceCollector) -> None:
        self.collector = collector

    def set_agent_data(self, epoch: int = None, num_simulated_games: int = None) -> None:
        if epoch is not None:
            self.epoch = epoch

        if num_simulated_games is not None:
            self.num_simulated_games = num_simulated_games

    def add_num_simulated_games(self, num: int) -> None:
        self.num_simulated_games += num

    def create_node(self, state: AbstractGameState, move_idx: int = None, parent: TreeNode = None) -> TreeNode:
        with torch.no_grad():
            state_tensor = self.encoder.encode(state)
            model_input = torch.tensor(np.expand_dims(state_tensor, axis=0), dtype=torch.float, device=self.device)
            prior, value = self.model(model_input)
            prior = prior[0].detach()
            value = value[0][0].detach()

        if self.noise and parent is None:
            noise_probs = np.random.dirichlet(self.alpha * np.ones(self.encoder.num_moves()))
            move_priors = {
                idx: 0.75 * p.item() + 0.25 * noise_probs[idx]
                for idx, p in enumerate(prior)
            }
        else:
            move_priors = {
                idx: p.item()
                for idx, p in enumerate(prior)
            }

        new_node = TreeNode(state, value, move_priors, parent, self.c, move_idx)

        if parent is not None:
            parent.add_child(move_idx, new_node)

        return new_node

    def select_branch(self, node: TreeNode) -> int:
        sqrt_total_n = np.sqrt(node.total_visit_count)
        branch_scores = node.get_scores(sqrt_total_n)

        return max(node.move_idxes(), key=lambda x: branch_scores[x])
        
    def select_branches(self, node: TreeNode, num: int):
        sqrt_total_n = np.sqrt(node.total_visit_count)
        branch_scores = node.get_scores(sqrt_total_n)

        return heapq.nlargest(num, node.move_idxes(), key=lambda x: branch_scores[x])

    @classmethod
    def get_value(cls, winner: Player) -> int:
        # 무승부인 경우
        if winner == Player.both:
            return 0
        else:
            return 1

    def select_move(self, game_state: AbstractGameState) -> Tuple[Move, Optional[np.ndarray]]:
        root = self.create_node(game_state)
        remain_rounds = self.simulations_per_move
        
        # 게임 진행
        for _ in range(remain_rounds):
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
    
    def simulate(self, root: TreeNode, next_move_idx: int, thread_lock):
        node = root
        # Selection
        while node.has_child(next_move_idx):
            node = node.get_child(next_move_idx)
            if not node.state.game_over:
                next_move_idx = self.select_branch(node)

        # Expansion & Simulation
        # 노드가 없거나 승자가 나왔으면 게임 진행 종료
        if node.state.game_over:
            move_idx = node.last_move_idx
            value = self.get_value(node.state.winner)
            node = node.parent
        else:
            next_move = self.encoder.decode_move_index(next_move_idx)
            new_state = node.state.apply_move(next_move)
            child_node = self.create_node(new_state, next_move_idx, parent=node)

            move_idx = next_move_idx
            if child_node.state.check_game_over():
                value = self.get_value(child_node.state.winner)
            else:
                value = -1 * child_node.value.item()

        # Backpropagation
        thread_lock.acquire()
        while node is not None:
            node.record_visit_and_update_score(move_idx, value)
            move_idx = node.last_move_idx
            node = node.parent
            value = -1 * value
        thread_lock.release()

    def train(self, dataset: ExperienceDataset, batch_size: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
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

    def save_agent(self, pthfile: str) -> None:
        state = {
            'encoder': self.encoder,
            'model': self.model,
            'num_simulated_games': self.num_simulated_games,
            'epoch': self.epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }

        torch.save(state, pthfile)

    @staticmethod
    def load_agent(pthfilename: str, device: str, num_threads_per_round: int = 1, noise: bool = False) -> SelfZeroAgent:
        loaded_file = torch.load(pthfilename, map_location='cpu')
        encoder = loaded_file['encoder']
        model = loaded_file['model']
        num_simulated_games = loaded_file['num_simulated_games']
        epoch = loaded_file['epoch']
        optimizer_state_dict = loaded_file['optimizer']
        scheduler_state_dict = loaded_file['scheduler']

        loaded_agent = ZeroAgent(encoder, model, device,
                                 num_threads_per_round=num_threads_per_round,
                                 noise=noise)
        loaded_agent.optimizer.load_state_dict(optimizer_state_dict)
        loaded_agent.scheduler.load_state_dict(scheduler_state_dict)
        loaded_agent.set_agent_data(epoch=epoch, num_simulated_games=num_simulated_games)
        print("simulated %d games." % num_simulated_games)
        return loaded_agent
