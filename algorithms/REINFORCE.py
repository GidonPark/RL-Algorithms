import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(nn.Module):
    def __init__(self,isEval, simulator, hidden_size, learning_rate, gamma, std=0.05):
        super(Agent, self).__init__()
        self.model_save_path = "./trained_model/REINFORCE.pt"
        self.isEval = isEval
        self.env = gym.make(simulator)
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.shape[0]
        self.output_limit = self.env.action_space.high[0]
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.rewards = []
        self.log_probs = []
        self.returns = []



        self.fc1 = nn.Linear(self.input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, self.output_dim)
        self.log_std = nn.Parameter(torch.ones(self.output_dim)*std)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.output_layer(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist

    def trajectory(self, reward, log_prob):
        self.rewards.append(reward)
        self.log_probs.append(log_prob)

    def model_save(self):
        torch.save({
            'actor_state_dict': self.state_dict(),
        }, self.model_save_path)

    def model_load(self):
        checkpoint = torch.load(self.model_save_path)
        self.load_state_dict(checkpoint['actor_state_dict'])

    def train(self):
        self.rewards.reverse()
        self.log_probs.reverse()
        R = 0
        self.optimizer.zero_grad()
        for r, action in zip(self.rewards, self.log_probs):
            R = r + self.gamma * R
            self.returns.append(R)
        self.returns = np.array(self.returns)
        self.returns = (self.returns - self.returns.mean()) / (self.returns.std() + 1e-5)
        loss = - self.dot_mean(self.returns, self.log_probs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.rewards = []
        self.log_probs = []
        self.returns = []

    def run(self, num_episode):
        if self.isEval :
            self.model_load()
        for i in range(num_episode):
            state = self.env.reset()
            state = torch.from_numpy(state).to(device).float()
            done = False
            step = 0
            rewards = 0
            while not done:
                self.env.render()
                if self.isEval:
                    dist = self.forward(state)
                    action = dist.sample()
                    action_excution = torch.tanh(action) * self.output_limit
                    next_state, reward, done, info = self.env.step(action_excution.cpu().data.numpy())
                else:
                    dist = self.forward(state)
                    action = dist.sample()
                    action_excution = torch.tanh(action) * self.output_limit
                    next_state, reward, done, info = self.env.step(action_excution.cpu().data.numpy())
                    log_prob = dist.log_prob(action).sum(dim=-1)
                    self.trajectory(reward, log_prob)
                next_state = torch.from_numpy(next_state).to(device).float()
                state = next_state
                rewards += reward.item()
                step += 1
            print('episode', i, 'step', step, 'rewards', rewards)

            if not self.isEval:
                self.train()
                if i % 2000 == 0:
                    self.model_save()


    def dot_mean(self, x, y):
        return sum(x_i * y_i for x_i, y_i in zip(x, y)) / len(x)










