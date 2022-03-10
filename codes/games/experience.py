from typing import Iterable, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from games.abstract_game_state import AbstractGameState


class ExperienceCollector:
    def __init__(self, board_size: int):
        """[summary]
            Collect status of game.
            Use with zero_agent.
        """
        self.board_size = board_size
        self.states = []
        self.visit_counts = []
        self.rewards = []
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def begin_episode(self) -> None:
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def record_decision(self, state: AbstractGameState, visit_counts: np.ndarray) -> None:
        self._current_episode_states.append(state)
        self._current_episode_visit_counts.append(visit_counts)

    def complete_episode(self, reward: float) -> None:
        # states = self._current_episode_states
        # visit_counts = self._current_episode_visit_counts
        states, visit_counts = self.episode_augmentation()
        num_states = len(states)
        self.states += states
        self.visit_counts += visit_counts
        self.rewards += [reward for _ in range(num_states)]

        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def episode_augmentation(self) -> Tuple[AbstractGameState, Iterable[np.ndarray]]:
        # rotate
        new_states = []
        new_visit_counts = []
        for state, visit_count in zip(self._current_episode_states, self._current_episode_visit_counts):
            for i in range(4):
                rotated_state = np.rot90(state, i, axes=(1, 2))
                rotated_visit_count = np.rot90(visit_count.reshape(self.board_size, self.board_size), i)
                new_states.append(rotated_state)
                new_visit_counts.append(rotated_visit_count.flatten())

                new_states.append(np.flip(rotated_state, 2))
                new_visit_counts.append(np.fliplr(rotated_visit_count).flatten())
        return new_states, new_visit_counts


class ExperienceDataset(Dataset):
    def __init__(self, board_size: int, num_planes: int, max_size: int):
        """[summary]
            This class returns experiences to train DeepLearning model.
            use add_experience to append experience.
        Args:
            board_size (int): Size of board.
            num_planes (int): Num of encoder planes.
            max_size (int): Max num of experiences.
        """
        self.board_size = board_size
        self.num_planes = num_planes
        self.max_size = max_size
        self.front = 0
        self.rear = 0

        self.state_memory = torch.zeros([self.max_size, num_planes, board_size, board_size])
        self.visit_count_memory = torch.zeros([self.max_size, board_size**2])
        self.visit_count_sum_memory = torch.zeros(self.max_size)
        self.reward_memory = torch.zeros(self.max_size)

    def __len__(self) -> int:
        """[summary]
            Returns num of experiences in this dataset.
        Returns:
            int: num of experiences.
        """
        return (self.rear - self.front + self.max_size) % self.max_size

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """[summary]
        Args:
            idx (int): [description]

        Returns:
            [type]: [description]
        """
        return [self.state_memory[idx],
                self.visit_count_memory[idx] / self.visit_count_sum_memory[idx],
                self.reward_memory[idx]]

    def add_experiences(self, collectors: Iterable[ExperienceCollector]) -> None:
        combined_states = torch.cat([torch.tensor(c.states, dtype=torch.float) for c in collectors])
        combined_visit_counts = torch.cat([torch.tensor(c.visit_counts) for c in collectors])
        combined_visit_sums = torch.sum(combined_visit_counts, dim=1)
        combined_rewards = torch.cat([torch.tensor(c.rewards) for c in collectors])

        for i in range(combined_states.size(0)):
            self.state_memory[self.rear] = (combined_states[i])
            self.visit_count_memory[self.rear] = (combined_visit_counts[i])
            self.visit_count_sum_memory[self.rear] = (combined_visit_sums[i])
            self.reward_memory[self.rear] = (combined_rewards[i])
            if self.front == (self.rear + 1) % self.max_size:
                self.front = (self.front + 1) % self.max_size
            self.rear = (self.rear + 1) % self.max_size
