import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
import gym
from algorithms.utils import Actor, ReplayBuffer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(nn.Module):
    def __init__(self, isEval, simulator, hidden_size, learning_rate, gamma, capacity, batch_size ,behavior_update ,std=0.05):
        super(Agent, self).__init__()
        self.model_save_path = "./trained_model/OffPG.pt"
        self.isEval = isEval
        self.env = gym.make(simulator)
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.shape[0]
        self.output_limit = self.env.action_space.high[0]
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.behavior_update = behavior_update
        self.step = 0
        self.tau = 1e-3
        self.epochs = 10
        self.batch_size = batch_size


        self.memory = ReplayBuffer(capacity)
        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward', 'done'))

        self.target_policy = Actor(self.input_dim, self.hidden_size, self.output_dim, std=std).to(device)
        self.behavior_policy = Actor(self.input_dim, self.hidden_size, self.output_dim, std=std).to(device)
        self.behavior_policy.load_state_dict(self.target_policy.state_dict())

        self.optimizer = optim.Adam(self.target_policy.parameters(), lr=self.learning_rate)

    def soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def model_save(self):
        torch.save({
            'actor_state_dict': self.behavior_policy.state_dict()
        }, self.model_save_path)

    def model_load(self):
        checkpoint = torch.load(self.model_save_path)
        self.behavior_policy.load_state_dict(checkpoint['actor_state_dict'])

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = self.Transition(*zip(*transitions))

        # Convert to tensor
        states = torch.cat(batch.state).reshape(-1,self.input_dim)
        rewards = torch.cat(batch.reward).float()
        actions = torch.cat(batch.action).reshape(-1, self.output_dim)
        dones = torch.cat(batch.done)

        returns = []

        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - done.int())
            returns.append(R)
        returns = reversed(torch.FloatTensor(returns).to(device))
        returns = returns - returns.mean() / (returns.std() + 1e-5)

        # Calculate log_probs of target policy
        for i in range(self.epochs):
            dists = self.target_policy(states)
            log_probs_target = dists.log_prob(actions).sum(dim=1)
            dists_behavior = self.behavior_policy(states)
            log_probs_behavior = dists_behavior.log_prob(actions).sum(dim=1).detach()

            # Calculate loss function
            importance_weights = torch.exp(log_probs_target - log_probs_behavior)
            loss = - (importance_weights * returns * log_probs_target).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.soft_update(self.target_policy, self.behavior_policy)

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
                    dist = self.behavior_policy(state)
                    action = dist.sample()
                    action_excution = torch.tanh(action) * self.output_limit
                    next_state, reward, done, info = self.env.step(action_excution.cpu().data.numpy())
                    next_state = torch.from_numpy(next_state).to(device).float()
                else:
                    dist = self.behavior_policy(state)
                    action = dist.sample()
                    action_excution = torch.tanh(action) * self.output_limit
                    next_state, reward, done, info = self.env.step(action_excution.cpu().data.numpy())
                    next_state = torch.from_numpy(next_state).to(device).float()
                    reward = torch.tensor([reward], device=device)
                    done = torch.tensor([done], device=device)
                    self.memory.push(state, action , next_state, reward, done)
                    self.step += 1
                total_rewards += reward.item()
                state = next_state
                step += 1
            print('episode', i, 'step', step, 'total_rewards', total_rewards)
            if not self.isEval :
                self.train()
                if i % 2000 == 0: self.model_save()


