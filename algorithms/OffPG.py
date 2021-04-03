import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from collections import namedtuple
import gym
from algorithms.utils import Actor, Critic
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(nn.Module):
    def __init__(self, simulator, hidden_size, learning_rate, gamma, behavior_update ,std=0.05):
        super(Agent, self).__init__()
        self.env = gym.make(simulator)
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.shape[0]
        self.output_limit = self.env.action_space.high[0]
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.behavior_update = behavior_update

        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'log_prob', 'done'))
        self.memory = []

        self.target_policy = Actor(self.input_dim, self.hidden_size, self.output_dim, std=std).to(device)
        self.behavior_policy = Actor(self.input_dim, self.hidden_size, self.output_dim, std=std).to(device)
        self.behavior_policy.load_state_dict(self.target_policy.state_dict())

        self.optimizer = optim.Adam(self.target_policy.parameters(), lr=self.learning_rate)


    def trajectory(self, *args):
        self.memory.append(self.Transition(*args))

    def train(self):
        transitions = self.Transition(*zip(*self.memory))

        # Convert to tensor
        states = torch.cat(transitions.state).reshape(-1,self.input_dim)
        rewards = torch.cat(transitions.reward).float()
        actions = torch.cat(transitions.action).reshape(-1, self.output_dim)
        log_probs_behavior = torch.cat(transitions.log_prob)
        dones = torch.cat(transitions.done)

        returns = []

        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - done.int())
            returns.append(R)
        returns = reversed(torch.FloatTensor(returns).to(device))
        returns = returns - returns.mean() / (returns.std() + 1e-5)

        # Calculate log_probs of target policy
        dists = self.target_policy(states)
        log_probs_target = dists.log_prob(actions).sum(dim=1)

        # Calculate loss function
        importance_weights = torch.exp(log_probs_target - log_probs_behavior).detach()
        loss = - (importance_weights * returns * log_probs_target).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []

    def run(self, num_episode):
        for i in range(num_episode):
            state = self.env.reset()
            state = torch.from_numpy(state).to(device).float()
            done = False
            step = 0
            total_rewards = 0
            while not done:
                self.env.render()
                dist = self.behavior_policy(state)
                action = dist.sample()
                action_excution = torch.tanh(action) * self.output_limit
                next_state, reward, done, info = self.env.step(action_excution.cpu().data.numpy())
                next_state = torch.from_numpy(next_state).to(device).float()
                log_prob = dist.log_prob(action).sum().unsqueeze(dim=0)
                total_rewards += reward
                reward = torch.tensor([reward], device=device)
                done = torch.tensor([done], device=device)
                self.trajectory(state, action , reward, log_prob, done)
                state = next_state
                step += 1
            print('episode', i, 'step', step, 'total_rewards', total_rewards)
            self.train()

            if i % self.behavior_update == 0:
                self.behavior_policy.load_state_dict(self.target_policy.state_dict())
