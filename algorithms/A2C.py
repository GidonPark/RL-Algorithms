import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from collections import namedtuple
import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(nn.Module):
    def __init__(self, simulator, hidden_size, learning_rate, gamma, std=0.0):
        super(Agent, self).__init__()
        self.env = gym.make(simulator)
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.shape[0]
        self.output_limit = self.env.action_space.high[0]
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.Transition = namedtuple('Transition', ('value', 'next_state', 'reward', 'log_prob', 'done'))
        self.memory = []

        self.fc1 = nn.Linear(self.input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.policy_layer = nn.Linear(hidden_size, self.output_dim)
        self.value_layer = nn.Linear(hidden_size, 1)
        #self.log_std = nn.Parameter(torch.ones(self.output_dim)*std)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.policy_layer(x)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        # std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        value = self.value_layer(x)
        return dist, value

    def trajectory(self, *args):
        self.memory.append(self.Transition(*args))

    def train(self):
        transitions = self.Transition(*zip(*self.memory))

        next_states = torch.cat(transitions.next_state).reshape(-1,self.input_dim)
        values = torch.cat(transitions.value)
        log_probs = torch.cat(transitions.log_prob)
        rewards = torch.cat(transitions.reward).float()
        dones = torch.cat(transitions.done)

        _, next_values = self.forward(next_states)

        # TD-Targets
        q_values = rewards + self.gamma * next_values.squeeze(dim=-1) * (1-dones.int())

        # Advantages
        advantages = q_values - values

        policy_loss = (- log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, q_values.detach())
        print('value loss', value_loss, 'policy_loss', policy_loss)
        total_loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        total_loss.backward()
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
                dist, value = self.forward(state)
                action = dist.sample()
                action_excution = torch.tanh(action) * self.output_limit
                next_state, reward, done, info = self.env.step(action_excution.cpu().data.numpy())
                next_state = torch.from_numpy(next_state).to(device).float()
                log_prob = dist.log_prob(action).sum().unsqueeze(dim=0)
                total_rewards += reward
                reward = torch.tensor([reward], device=device)
                done = torch.tensor([done], device=device)
                self.trajectory(value, next_state, reward, log_prob, done)
                state = next_state
                step += 1
            print('episode', i, 'step', step, 'total_rewards', total_rewards)

            self.train()

