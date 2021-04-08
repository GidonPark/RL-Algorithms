import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions import Normal
from collections import namedtuple
import random, math
import numpy as np

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, std=1):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_dim)
        self.log_std = nn.Parameter(torch.ones(output_dim) * std)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.output_layer(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        nn.init.uniform_(self.fc1.weight.data, a=- 1 / math.sqrt(input_dim), b=1 / math.sqrt(input_dim))
        nn.init.uniform_(self.fc2.weight.data, a=- 1 / math.sqrt(hidden_size), b=1 / math.sqrt(hidden_size))
        nn.init.uniform_(self.output_layer.weight.data, a = -3e-3, b=3e-3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.output_layer(x)
        return value


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[int(self.position)] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.15, 0.01, 0.2
        self.mu = mu
        self.x_prev = torch.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)
