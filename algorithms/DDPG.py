import torch, gym, math
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from collections import namedtuple


from algorithms.utils import ReplayBuffer, OrnsteinUhlenbeckNoise, fanin_init
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc_s_1 = nn.Linear(state_dim, hidden_size)
        self.fc_s_2 = nn.Linear(hidden_size, hidden_size)
        self.fc_a = nn.Linear(action_dim, hidden_size)
        self.fc_q = nn.Linear(2 * hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.init_weigths(init_w)

    def init_weigths(self, init_w):
        self.fc_s_1.weight.data = fanin_init(self.fc_s_1.weight.data.size())
        self.fc_s_2.weight.data = fanin_init(self.fc_s_2.weight.data.size())
        self.fc_q.weight.data = fanin_init(self.fc_q.weight.data.size())
        self.output_layer.weight.data.uniform_(-init_w, init_w)

    def forward(self, x, a):
        x = F.relu(self.fc_s_1(x))
        x = F.relu(self.fc_s_2(x))
        y = F.relu(self.fc_a(a))
        q = F.relu(self.fc_q(torch.cat([x, y], dim=1)))
        q = self.output_layer(q)
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
        self.model_save_path = "./trained_model/DDPG.pt"
        self.isEval = isEval
        self.env = gym.make(simulator)
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.shape[0]
        self.output_limit = self.env.action_space.high[0]
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.behavior_update = behavior_update
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3
        self.tau = 1e-3


        self.mu = DeterministicActor(self.input_dim, self.hidden_size, self.output_dim).to(device)
        self.mu_target = DeterministicActor(self.input_dim, self.hidden_size, self.output_dim).to(device)
        self.critic = Critic(self.input_dim, self.output_dim, self.hidden_size).to(device)
        self.critic_target = Critic(self.input_dim, self.output_dim, self.hidden_size).to(device)

        # Copy weights
        self.mu_target.load_state_dict(self.mu.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        #Optimizers
        self.actor_optimizer = optim.Adam(self.mu.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr, weight_decay=1e-2)

        #Exploration noise process
        self.noise = OrnsteinUhlenbeckNoise(torch.zeros(self.output_dim))

        self.memory = ReplayBuffer(capacity)
        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward', 'done'))

    def soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def model_save(self):
        torch.save({
            'actor_state_dict': self.mu.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, self.model_save_path)

    def model_load(self):
        checkpoint = torch.load(self.model_save_path)
        self.mu.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])


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

        #TD-target
        td_target = rewards + self.gamma * self.critic_target(next_states, self.mu_target(next_states)).squeeze(dim=-1) * (1-dones.int())
        td_loss = F.smooth_l1_loss(self.critic(states, actions).squeeze(dim=-1), td_target.detach())
        self.critic_optimizer.zero_grad()
        td_loss.backward()
        self.critic_optimizer.step()

        mu_loss = - self.critic(states,self.mu(states)).mean()
        self.actor_optimizer.zero_grad()
        mu_loss.backward()
        self.actor_optimizer.step()


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
                    action = self.mu(state) + torch.FloatTensor(0.5 / math.log(i+2) * np.random.randn(self.output_dim)).to(device)
                    next_state, reward, done, info = self.env.step(np.clip(self.output_limit * action.cpu().data.numpy(), -self.output_limit, self.output_limit))
                    next_state = torch.from_numpy(next_state).to(device).float()
                    reward = torch.tensor([reward], device=device)
                    done = torch.tensor([done], device=device)
                    self.memory.push(state, action , next_state, reward, done)

                total_rewards += reward.item()
                state = next_state
                step += 1
            print('episode', i, 'step', step, 'total_rewards', total_rewards)

            if not self.isEval:
                for j in range(10):
                    self.train()
                    self.soft_update(self.mu, self.mu_target)
                    self.soft_update(self.critic, self.critic_target)
                if i % 2000 == 0:
                    self.model_save()


