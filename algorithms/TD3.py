import torch, gym, math
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from collections import namedtuple


from algorithms.utils import ReplayBuffer, fanin_init
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim+action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.init_weigths(init_w)

    def init_weigths(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.output_layer.weight.data.uniform_(-init_w, init_w)

    def forward(self, x, a):
        x = torch.cat([x,a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.output_layer(x)
        return q

class DeterministicActor(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, init_w=3e-3):
        super(DeterministicActor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_dim)
        self.init_weigths(init_w)

    def init_weigths(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.output_layer.weight.data.uniform_(-init_w, init_w)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.output_layer(x))
        return mu


class Agent(nn.Module):
    def __init__(self, isEval, simulator, hidden_size, gamma, behavior_update, capacity, batch_size):
        super(Agent, self).__init__()
        self.model_save_path = "./trained_model/TD3.pt"
        self.isEval = isEval
        self.env = gym.make(simulator)
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.shape[0]
        self.output_limit = self.env.action_space.high[0]
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.behavior_update = behavior_update
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4
        self.tau = 5e-3
        self.step = 0
        self.delay_param = 2


        self.mu = DeterministicActor(self.input_dim, self.hidden_size, self.output_dim).to(device)
        self.q1 = Critic(self.input_dim, self.output_dim, self.hidden_size).to(device)
        self.q2 = Critic(self.input_dim, self.output_dim, self.hidden_size).to(device)

        # Target networks
        self.mu_target = DeterministicActor(self.input_dim, self.hidden_size, self.output_dim).to(device)
        self.q1_target = Critic(self.input_dim, self.output_dim, self.hidden_size).to(device)
        self.q2_target = Critic(self.input_dim, self.output_dim, self.hidden_size).to(device)

        # Copy weights
        self.mu_target.load_state_dict(self.mu.state_dict())
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        #Optimizers
        self.qf_parameters = list(self.q1.parameters()) + list(self.q2.parameters())
        self.actor_optimizer = optim.Adam(self.mu.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.qf_parameters, lr=self.critic_lr, weight_decay=1e-2)

        self.memory = ReplayBuffer(capacity)
        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward', 'done'))

    def soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def model_save(self):
        torch.save({
            'actor_state_dict': self.mu.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
        }, self.model_save_path)

    def model_load(self):
        checkpoint = torch.load(self.model_save_path)
        self.mu.load_state_dict(checkpoint['actor_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])


    def train(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = self.Transition(*zip(*transitions))

        # Convert to tensor
        states = torch.cat(batch.state).reshape(-1,self.input_dim).float()
        rewards = torch.cat(batch.reward).float()
        actions = torch.cat(batch.action).reshape(-1, self.output_dim).float().detach()
        next_states = torch.cat(batch.next_state).reshape(-1,self.input_dim).float()
        dones = torch.cat(batch.done)

        with torch.no_grad():
            epsilon = torch.clamp(torch.normal(mean=0.0, std=0.2, size=(1, self.output_dim)).squeeze(dim=0), -0.5,
                                  0.5).to(device)
            action_tilt = self.mu_target(next_states) + epsilon
            q1 = self.q1_target(next_states, action_tilt)
            q2 = self.q2_target(next_states, action_tilt)

            # TD-target
            td_target = rewards + self.gamma * torch.min(q1, q2).squeeze(dim=-1) * (1 - dones.int())

        q1_td_loss = F.mse_loss(self.q1(states, actions).squeeze(dim=-1), td_target)
        q2_td_loss = F.mse_loss(self.q2(states, actions).squeeze(dim=-1), td_target)
        td_loss = q1_td_loss + q2_td_loss
        self.critic_optimizer.zero_grad()
        td_loss.backward()
        self.critic_optimizer.step()

        if self.step % self.delay_param == 0:
            mu_loss = - self.q1(states, self.mu(states)).mean()
            self.actor_optimizer.zero_grad()
            mu_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.mu, self.mu_target)
            self.soft_update(self.q1, self.q1_target)
            self.soft_update(self.q2, self.q2_target)




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
                    action = self.mu(state)
                    next_state, reward, done, info = self.env.step(self.output_limit * action.cpu().data.numpy())
                    next_state = torch.from_numpy(next_state).to(device).float()
                else:
                    action = self.mu(state) + torch.normal(mean=0.0, std=0.2, size=(1, self.output_dim)).squeeze(dim=0).to(device)
                    next_state, reward, done, info = self.env.step(np.clip(self.output_limit * action.cpu().data.numpy(), -self.output_limit, self.output_limit))
                    next_state = torch.from_numpy(next_state).to(device).float()
                    reward = torch.tensor([reward], device=device)
                    done = torch.tensor([done], device=device)
                    self.memory.push(state, action, next_state, reward, done)
                    self.train()
                    self.step += 1

                total_rewards += reward.item()
                state = next_state
                step += 1
            print('episode', i, 'step', step, 'total_rewards', total_rewards)

            if not self.isEval and i % 2000 == 0: self.model_save()


