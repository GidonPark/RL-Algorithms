import argparse
import gym
import torch

parser = argparse.ArgumentParser(description='Mujoco simulation with Reinforcement Learning')
parser.add_argument('--mode', type=str, default='train', help='train, test')
parser.add_argument('--simulator', type=str, default='Hopper-v2', help='simulator')
parser.add_argument('--num-episode', type=int, default=1000000, help='number of episode')
parser.add_argument('--gamma', type=float, default=0.99, help='discount fator')
parser.add_argument('--algorithms'
                    , type=str, default='REINFORCE', help='algorithm')
parser.add_argument('--learning-rate', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--hidden-size', type=int, default=64, help='Neurons in each layer')
args = parser.parse_args()

env = gym.make(args.simulator)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Make agent
if args.algorithms == 'REINFORCE':
    from algorithms import REINFORCE
    agent = REINFORCE.Agent(args.simulator, args.hidden_size, args.learning_rate, args.gamma).to(device)
elif args.algorithms == 'A2C':
    from algorithms import A2C
    agent = A2C.Agent(args.simulator, args.hidden_size, args.learning_rate, args.gamma).to(device)
elif args.algorithms == 'A2C_v2':
    from algorithms import A2C_v2
    agent = A2C_v2.Agent(args.simulator, args.hidden_size, args.learning_rate, args.gamma).to(device)

agent.run(args.num_episode)







