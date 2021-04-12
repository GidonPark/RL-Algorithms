from algorithms.utils import Actor, Critic
import torch.nn as nn
import torch, gym, math
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from collections import namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(nn.Module):
    def __init__(self, isEval, simulator, hidden_size, gamma, lamb):
        super(Agent, self).__init__()
        self.model_save_path = "./trained_model/PPO.pt"
        self.isEval = isEval
        self.env = gym.make(simulator)
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.shape[0]
        self.output_limit = self.env.action_space.high[0]
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.lamb = lamb
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4
        self.tau = 1e-3
        self.epochs = 10
        self.epsilon = 0.2


        self.policy = Actor(self.input_dim, self.hidden_size, self.output_dim).to(device)
        self.critic = Critic(self.input_dim, self.hidden_size, self.output_dim).to(device)

        #Optimizers
        self.actor_optimizer = optim.Adam(self.policy.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        #Exploration noise process
        self.memory = []
        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state' ,'reward', 'done', 'log_prob'))


    def trajectory(self, *args):
        self.memory.append(self.Transition(*args))


    def model_save(self):
        torch.save({
            'actor_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, self.model_save_path)

    def model_load(self):
        checkpoint = torch.load(self.model_save_path)
        self.policy.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])


    def train(self):
        transitions = self.Transition(*zip(*self.memory))
        # Convert to tensor
        states = torch.cat(transitions.state).reshape(-1,self.input_dim).float()
        rewards = torch.cat(transitions.reward).float()
        actions = torch.cat(transitions.action).reshape(-1, self.output_dim).float().detach()
        next_states = torch.cat(transitions.next_state).reshape(-1,self.input_dim).float()
        dones = torch.cat(transitions.done)
        log_probs_old = torch.cat(transitions.log_prob)

        for i in range(self.epochs):
            td_target = rewards + self.gamma * self.critic(next_states).squeeze(dim=-1) * (1-dones.int())
            deltas = td_target - self.critic(states).squeeze(dim=-1)
            deltas = deltas.detach().cpu().numpy()

            advantages = []
            advantage = 0
            for delta in deltas[::-1]:
                advantage = self.gamma * self.lamb * advantage + delta
                advantages.append(advantage)

            advantages.reverse()
            advantages = torch.FloatTensor(advantages).to(device)

            dist = self.policy(states)
            log_probs = dist.log_prob(actions).sum(dim=1)
            ratio = torch.exp(log_probs - log_probs_old)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1+self.epsilon, 1-self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.smooth_l1_loss(self.critic(states).squeeze(dim=-1), td_target.detach())
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
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
                    dist = self.policy(state)
                    action = dist.sample()
                    next_state, reward, done, info = self.env.step(np.clip(action.cpu().data.numpy(), -self.output_limit, self.output_limit))
                    next_state = torch.from_numpy(next_state).to(device).float()
                else:
                    dist = self.policy(state)
                    action = dist.sample()
                    next_state, reward, done, info = self.env.step(np.clip(action.cpu().data.numpy(), -self.output_limit, self.output_limit))
                    next_state = torch.from_numpy(next_state).to(device).float()
                    reward = torch.tensor([reward], device=device)
                    done = torch.tensor([done], device=device)
                    log_prob = dist.log_prob(action).sum().unsqueeze(dim=0).detach()
                    self.trajectory(state, action, next_state, reward, done, log_prob)


                total_rewards += reward.item()
                state = next_state
                step += 1
            print('episode', i, 'step', step, 'total_rewards', total_rewards)

            if not self.isEval:
                self.train()
                if i % 2000 == 0:
                    self.model_save()
