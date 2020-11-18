from typing import (
    Tuple,
)

import torch

from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    TensorStack5,
    TorchDevice,
)

import random
import numpy as np

class ReplayMemory(object):

    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,
    ) -> None:
        self.__device = device
        self.__capacity = capacity
        self.__size = 0
        self.__pos = 0

        self.__m_states = torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8)
        self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)

    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
    ) -> None:
        self.__m_states[self.__pos] = folded_state
        self.__m_actions[self.__pos, 0] = action
        self.__m_rewards[self.__pos, 0] = reward
        self.__m_dones[self.__pos, 0] = done

        self.__pos = (self.__pos + 1) % self.__capacity
        self.__size = max(self.__size, self.__pos)

    def sample(self, batch_size: int) -> Tuple[
            BatchState,
            BatchAction,
            BatchReward,
            BatchNext,
            BatchDone,
    ]:
        indices = torch.randint(0, high=self.__size, size=(batch_size,))
        b_state = self.__m_states[indices, :4].to(self.__device).float()
        b_next = self.__m_states[indices, 1:].to(self.__device).float()
        b_action = self.__m_actions[indices].to(self.__device)
        b_reward = self.__m_rewards[indices].to(self.__device).float()
        b_done = self.__m_dones[indices].to(self.__device).float()
        return b_state, b_action, b_reward, b_next, b_done

    def __len__(self) -> int:
        return self.__size


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class MemoryBufferPER(object):
    # stored as ( s, a, r, s_ ) in SumTree
    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,
            a=0.6, e=0.01
    ) -> None:
        self.__device = device
        self.__capacity = capacity
        self.__size = 0
        self.__pos = 0

        self.__m_states = torch.zeros(
            (capacity, channels, 84, 84), dtype = torch.uint8)
        self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)

        self.tree = SumTree(capacity)
        self.memory_size = capacity
        self.prio_max = 0.1
        self.a = a
        self.e = e

    def push(self, state, action, reward, done):
        data = (state, action, reward, done)
        p = (np.abs(self.prio_max) + self.e) ** self.a  # proportional priority
        self.tree.add(p, data)

    def sample(self, batch_size):
        idxs = []
        pos = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)

            state, action, reward, done = data
            self.__m_states[self.__pos] = state
            self.__m_actions[self.__pos, 0] = action
            self.__m_rewards[self.__pos, 0] = reward
            self.__m_dones[self.__pos, 0] = done

            pos.append(self.__pos)
            self.__pos = (self.__pos + 1) % self.__capacity
            priorities.append(p)
            idxs.append(idx)

        indices = torch.tensor(pos)
        b_state = self.__m_states[indices, :4].to(self.__device).float()
        b_next = self.__m_states[indices, 1:].to(self.__device).float()
        b_action = self.__m_actions[indices].to(self.__device)
        b_reward = self.__m_rewards[indices].to(self.__device).float()
        b_done = self.__m_dones[indices].to(self.__device).float()
        return idxs, b_state, b_action, b_reward, b_next, b_done

    def update(self, idxs, errors):
        self.prio_max = max(self.prio_max, max(np.abs(errors)))
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.e) ** self.a
            self.tree.update(idx, p)

    def size(self):
        return self.tree.n_entries

    def __len__(self) -> int:
        return self.size()
