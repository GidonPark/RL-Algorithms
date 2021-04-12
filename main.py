import argparse
import gym
import torch

parser = argparse.ArgumentParser(description='Mujoco simulation with Reinforcement Learning')
parser.add_argument('--isEval', type=bool, default=False, help='train = (False) or test = (True)')
parser.add_argument('--simulator', type=str, default='Humanoid-v2', help='simulator')
parser.add_argument('--num-episode', type=int, default=100000, help='number of episode')
parser.add_argument('--gamma', type=float, default=0.99, help='discount fator')
parser.add_argument('--algorithms'
                    , type=str, default='PPO', help='algorithm')
parser.add_argument('--learning-rate', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--hidden-size', type=int, default=400, help='Neurons in each layer')
parser.add_argument('--behavior-update', type=int, default=10, help='Update behavior policy of Off-Policy methods')
parser.add_argument('--replay_buffer_size', type=int, default=5e4, help='Size of replay buffer')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--lamb', type=float, default=0.95, help='GAE parameter')
args = parser.parse_args()

env = gym.make(args.simulator)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Make agent
if args.algorithms == 'REINFORCE':
    from algorithms import REINFORCE
    agent = REINFORCE.Agent(args.isEval, args.simulator, args.hidden_size, args.learning_rate, args.gamma).to(device)
if args.algorithms == 'Off_Policy':
    from algorithms import OffPG
    agent = OffPG.Agent(args.isEval, args.simulator, args.hidden_size, args.learning_rate, args.gamma, args.behavior_update).to(device)
elif args.algorithms == 'A2C':
    from algorithms import A2C
    agent = A2C.Agent(args.isEval, args.simulator, args.hidden_size, args.learning_rate, args.gamma).to(device)
elif args.algorithms == 'A2C_v2':
    from algorithms import A2C_v2
    agent = A2C_v2.Agent(args.isEval, args.simulator, args.hidden_size, args.learning_rate, args.gamma).to(device)
elif args.algorithms == 'DDPG':
    from algorithms import DDPG
    agent = DDPG.Agent(args.isEval, args.simulator, args.hidden_size, args.gamma, args.behavior_update, args.replay_buffer_size, args.batch_size).to(device)
elif args.algorithms == 'PPO':
    from algorithms import PPO
    agent = PPO.Agent(args.isEval, args.simulator, args.hidden_size, args.gamma, args.lamb).to(device)

agent.run(args.num_episode)







