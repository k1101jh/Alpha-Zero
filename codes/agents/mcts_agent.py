import numpy as np
import threading
import random

from codes.agents.abstract_agent import Agent
from codes.game_types import Player


class TreeNode:
    def __init__(self, state, parent):
        self.state = state
        self.value = 0
        self.parent = parent
        self.visit_count = 0
        self.branches = []
        for idx in range(state.board.board_size * state.board.board_size):
            if self.state.check_valid_move_idx(idx):
                self.branches.append(idx)
        self.children = {}

    def move_idxes(self):
        return self.branches

    def add_child(self, move_idx, child_node):
        self.children[move_idx] = child_node

    def has_child(self, move_idx):
        return move_idx in self.children

    def get_child(self, move_idx):
        return self.children[move_idx]

    def record_visit(self, value):
        self.visit_count += 1
        self.value += value

    def expected_value(self, move_idx):
        if move_idx in self.branches:
            child = self.children[move_idx]
            if child.visit_count == 0:
                return 0.0
            return child.value / child.visit_count
        else:
            return 0.0


class MCTSAgent(Agent):
    def __init__(self, encoder, simulations_per_move=300, num_threads_per_round=12):
        """[summary]
            Use Monte Carlo Tree Search(MCTS) Algorithm to select move.
        Args:
            encoder ([type]): [description]
            simulations_per_move (int, optional): [description]. Defaults to 300.
            num_threads_per_round (int, optional): [description]. Defaults to 12.
        """
        super().__init__()
        self.encoder = encoder
        self.num_simulated_games = 0
        self.num_threads_per_round = num_threads_per_round
        self.simulations_per_move = simulations_per_move
        self.lock = threading.Lock()

    def add_num_simulated_games(self, num):
        self.num_simulated_games += num

    @classmethod
    def create_node(cls, state, move_idx=None, parent=None):
        new_node = TreeNode(state, parent)

        if parent is not None:
            parent.add_child(move_idx, new_node)

        return new_node

    @classmethod
    def select_branch(cls, node):
        return random.choice(node.move_idxes())

    @classmethod
    def select_branches(cls, node, num):
        return random.sample(node.move_idxes(), num)

    def select_move(self, game_state):
        root = self.create_node(game_state)
        remain_rounds = self.simulations_per_move
        # 게임 진행
        while remain_rounds > 0:
            num_thread = min(self.num_threads_per_round, remain_rounds)

            first_move_idx_candidates = self.select_branches(root, num_thread)
            threads = []
            for first_move_idx_candidate in first_move_idx_candidates:
                t = threading.Thread(target=self.simulate, args=(root, first_move_idx_candidate))
                t.daemon = True
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            remain_rounds -= num_thread

        expected_value = np.array([
            root.expected_value(idx)
            for idx in range(self.encoder.num_moves())
        ])

        selected_move_idx = max(root.move_idxes(), key=root.expected_value)
        return self.encoder.decode_move_index(selected_move_idx), expected_value

    def simulate(self, root, next_move_idx):
        node = root
        while not node.state.game_over:
            next_move = self.encoder.decode_move_index(next_move_idx)
            new_state = node.state.apply_move(next_move)
            if node.has_child(next_move_idx):
                node = node.get_child(next_move_idx)
            else:
                node = self.create_node(new_state, next_move_idx, node)
            if not node.state.check_game_over():
                next_move_idx = self.select_branch(node)

        # 승자가 없으면(돌을 더 놓을 수 없으면)
        if node.state.winner == Player.both:
            value = 0
        elif node.state.winner == root.state.player:
            value = 1
        else:
            value = -1

        node = node.parent
        move_idx = next_move_idx
        self.lock.acquire()
        while node is not None:
            node.record_visit(move_idx, value)
            move_idx = node.last_move_idx
            node = node.parent
            value = -1 * value
        self.lock.release()
