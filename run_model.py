import argparse
import os

import gymnasium as gym
import numpy as np
import torch

from dqn_common import epsilon_by_frame, DqnNetSingleLayer, DqnNetTwoLayers, alpha_sync, DuellingDqn
import yaml

# To run this function (assuming it is saved as run_model.py in the same directory as dqn_gym.py):
#  python run_model.py -n <network-type> -e <environment> -m <path-to-model>
#
# Note that the network type must matched the saved type that the model was originally trained on and
# also the same torch device must be used to run the model as the one it was originally trained on 
# as torch saves metadata about where the tensors were created and these can be remapped to another 
# device.

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--env", default="CartPole-v1", help="Full name of the environment, e.g. CartPole-v1, LunarLander-v2, etc.")
parser.add_argument("-c", "--config_file", default="config/dqn.yaml", help="Config file with hyper-parameters")
parser.add_argument("-n", "--network", default='s',
                    help="DQN network architecture `single-hidden` for single hidden layer, `two-hidden` for 2 hidden layers and `duelling-dqn` for duelling DQN",
                    choices=['single-hidden', 'two-hidden', 'duelling-dqn'])
parser.add_argument("-m", "--model", type=str, help="Path to model")
args = parser.parse_args()

# Hyperparameters for the requried environment
hypers = yaml.load(open(args.config_file), Loader=yaml.FullLoader)

env = gym.make(args.env, render_mode='human')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("Running on CPU")

if args.env not in hypers:
    raise Exception(f'Hyper-parameters not found for env {args.env} - please add it to the config file (config/dqn.yaml)')
params = hypers[args.env]

if args.network == 'two-hidden':
    net = DqnNetTwoLayers(obs_size=env.observation_space.shape[0],
                          hidden_size=params['hidden_size'], hidden_size2=params['hidden_size2'],
                          n_actions=env.action_space.n).to(device)
elif args.network == 'single-hidden':
    net = DqnNetSingleLayer(obs_size=env.observation_space.shape[0],
                            hidden_size=params['hidden_size'],
                            n_actions=env.action_space.n).to(device)
else:
    net = DuellingDqn(obs_size=env.observation_space.shape[0],
                      hidden_size=params['hidden_size'],
                      n_actions=env.action_space.n).to(device)

net.load_state_dict(torch.load(args.model))

state, _ = env.reset()
total_reward = 0
episodes = 0

while True:
    state_a = np.array([state], copy=False)
    state_v = torch.tensor(state_a).to(device)
    q_vals_v = net(state_v)
    _, act_v = torch.max(q_vals_v, dim=1)
    action = act_v.item()

    new_state, reward, terminated, truncated, _ = env.step(action)
    is_done = terminated or truncated

    state = new_state
    total_reward += reward

    if is_done:
        episodes += 1
        print(f"Episode {episodes} completed with total reward: {total_reward}")
        total_reward = 0
        state, _ = env.reset()
