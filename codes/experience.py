import numpy as np
import torch
from torch.utils.data import Dataset


class ExperienceCollector:
    def __init__(self):
        self.states = []
        self.visit_counts = []
        self.rewards = []
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def record_decision(self, state, visit_counts):
        self._current_episode_states.append(state)
        self._current_episode_visit_counts.append(visit_counts)

    def complete_episode(self, reward):
        states, visit_counts = self.episode_augmentation()
        num_states = len(states)
        self.states += states
        self.visit_counts += visit_counts
        self.rewards += [reward for _ in range(num_states)]

        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def episode_augmentation(self):
        # rotate
        board_size = self._current_episode_states[0][0].shape[0]
        new_states = []
        new_visit_counts = []
        for state, visit_count in zip(self._current_episode_states, self._current_episode_visit_counts):
            for i in range(4):
                new_states.append(np.rot90(state, i, axes=(1, 2)))
                new_visit_counts.append(np.rot90(visit_count.reshape(board_size, board_size), i).flatten())

            new_states.append(np.flip(state, 2))
            new_visit_counts.append(np.fliplr(visit_count.reshape(board_size, board_size)).flatten())
        return new_states, new_visit_counts


class ExperienceDataset(Dataset):
    def __init__(self, board_size, num_planes, max_size):
        self.max_size = max_size
        self.front = 0
        self.rear = 0

        self.state_memory = torch.zeros([self.max_size, num_planes, board_size, board_size])
        self.visit_count_memory = torch.zeros([self.max_size, board_size**2])
        self.visit_count_sum_memory = torch.zeros(self.max_size)
        self.reward_memory = torch.zeros(self.max_size)

    def __len__(self):
        return (self.rear - self.front + self.max_size) % self.max_size

    def __getitem__(self, idx):
        return [self.state_memory[idx],
                self.visit_count_memory[idx] / self.visit_count_sum_memory[idx],
                self.reward_memory[idx]]

    def add_experiences(self, collectors):
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

    def serialize(self, h5file):
        h5file.create_group('experience')
        h5file['experience'].create_dataset('state', data=self.state_memory)
        h5file['experience'].create_dataset('visit_count', data=self.visit_count_memory)
        h5file['experience'].create_dataset('visit_count_sum', data=self.visit_count_sum_memory)
        h5file['experience'].create_dataset('reward', data=self.reward_memory)

    @staticmethod
    def load(h5file):
        state_memory = h5file['experience']['state']
        visit_count_memory = h5file['experience']['visit_count']
        visit_count_sum_memory = h5file['experience']['visit_count_sum']
        reward_memory = h5file['experience']['reward']

        loaded_obj = ExperienceDataset()
        loaded_obj.state_memory = state_memory
        loaded_obj.visit_count_memory = visit_count_memory
        loaded_obj.visit_count_sum_memory = visit_count_sum_memory
        loaded_obj.reward_memory = reward_memory
        return loaded_obj
