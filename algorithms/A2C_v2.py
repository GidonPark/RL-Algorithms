import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
import gym
from algorithms.utils import Actor, Critic
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(nn.Module):
    def __init__(self, isEval, simulator, hidden_size, learning_rate, gamma, std=0.05):
        super(Agent, self).__init__()
        self.model_save_path = "./trained_model/A2C_splitVersion.pt"
        self.isEval = isEval
        self.env = gym.make(simulator)
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.shape[0]
        self.output_limit = self.env.action_space.high[0]
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.Transition = namedtuple('Transition', ('value', 'next_state', 'reward', 'log_prob', 'done'))
        self.memory = []

        self.actor = Actor(self.input_dim, self.hidden_size, self.output_dim, std=std)
        self.critic = Critic(self.input_dim, self.hidden_size)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.learning_rate)


    def trajectory(self, *args):
        self.memory.append(self.Transition(*args))

    def model_save(self):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, self.model_save_path)

    def model_load(self):
        checkpoint = torch.load(self.model_save_path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

    def train(self):
        transitions = self.Transition(*zip(*self.memory))

        next_states = torch.cat(transitions.next_state).reshape(-1,self.input_dim)
        values = torch.cat(transitions.value)
        log_probs = torch.cat(transitions.log_prob)
        rewards = torch.cat(transitions.reward).float()
        dones = torch.cat(transitions.done)

        next_values = self.critic(next_states)

        # TD-Targets
        q_values = rewards + self.gamma * next_values.squeeze(dim=-1) * (1-dones.int())

        # Advantages
        advantages = q_values - values

        policy_loss = (- log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, q_values.detach())
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        policy_loss.backward()
        value_loss.backward()
        self.actor_optim.step()
        self.critic_optim.step()
        self.memory = []

    def run(self, num_episode):
        if self.isEval :
            self.model_load()
        for i in range(num_episode):
            state = self.env.reset()
            state = torch.from_numpy(state).to(device).float()
            done = False
            step = 0
            total_rewards = 0
            while not done:
                self.env.render()
                if self.isEval:
                    dist = self.actor(state)
                    action = dist.sample()
                    action_excution = torch.tanh(action) * self.output_limit
                    next_state, reward, done, info = self.env.step(action_excution.cpu().data.numpy())
                    next_state = torch.from_numpy(next_state).to(device).float()
                else:
                    dist = self.actor(state)
                    value = self.critic(state)
                    action = dist.sample()
                    action_excution = torch.tanh(action) * self.output_limit
                    next_state, reward, done, info = self.env.step(action_excution.cpu().data.numpy())
                    next_state = torch.from_numpy(next_state).to(device).float()
                    log_prob = dist.log_prob(action).sum().unsqueeze(dim=0)
                    reward = torch.tensor([reward], device=device)
                    done = torch.tensor([done], device=device)
                    self.trajectory(value, next_state, reward, log_prob, done)
                total_rewards += reward.item()
                state = next_state
                step += 1
            print('episode', i, 'step', step, 'total_rewards', total_rewards)

            if not self.isEval:
                self.train()
                if i % 2000 == 0:
                    self.model_save()