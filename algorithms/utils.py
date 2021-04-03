import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, std=1):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_dim)
        self.log_std = nn.Parameter(torch.ones(output_dim) * std)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.output_layer(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.output_layer(x)
        return value