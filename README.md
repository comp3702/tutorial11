# Basic DQN

This repo has some basic DQN and Duelling DQN examples.

### Requirements
I use conda to manage virtual environments so you will need [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (or just install all the packages manually).
To install the dependencies with conda use:

    conda env create -f environment.yml

This way, PyTorch without GPU will be installed. If you have a GPU and want a GPU version, follow [these instructions](https://pytorch.org/get-started/locally/).

### Running

There are 2 versions: [dqn_cartpole.py](dqn_cartpole.py) is a simple implementation with all the parameters hardcoded 
and [dqn_gym.py](dqn_gym.py) where the hyper-parameters are in the external [config file](config/dqn.yaml), and it also provides implementation for a single hidden layer DQN, two hidden layers DQN, as well as the Duelling DQN.

To run the simple example:

    python dqn_cartpole.py

This will train on `CartPole-v0`, which is a deprecated version of the environment, but it's much easier to solve.
With the default hyper-params it should start learning at about 13k frames and it should reach R100 of 195 at about 40k.

#### [OpenAI Gym](https://www.gymlibrary.dev/) Environments

[dqn_gym.py](dqn_gym.py) has the hyper-params are moved to the [config file](config/dqn.yaml). The config file has 3 sets of parameters, that are selected based on the environment (there is `CartPole-v0`, `CartPole-v1` and `LunarLander-v2`), controlled by command line argument `-e ...`.

The DQN network structure is controlled by command line argument `-n ` 
(with choices of `single-hidden` for a single hidden layer, `two-hidden` for 2 hidden layers (both for a plain [DQN](https://arxiv.org/abs/1312.5602)), and `duelling-dqn` for [Duelling DQN](https://arxiv.org/abs/1511.06581)).

It also saves the trained model in [saved_models/](saved_models/).

    python dqn_gym.py -e CartPole-v1 -n two-hidden
    python dqn_gym.py -e CartPole-v1 -n duelling-dqn

`CartPole-v1` is much more difficult to solve than `CartPole-v0`, but both `d` and `dd` options should do it, although the number of frames may vary greatly.
The solved threshold is 475 (which means the R100 needs to reach 475). Both of them also use soft updates of the target network, rather than full update every 1k frames.

__The visualizer will open when the R100 reaches 95%.__

This is how it looks when it's almost solved:

![CartPole-v1](resources/CartPole-v1.gif)

Values in `R100` column are ~447, and the immediate reward in `R: ` is mostly 500 (the highest possible reward). Column `Epsilon` shows the decaying value of exploration.
The pendulum remains upgright for most of the episode.

#### LunarLander-v2 Environment

    python dqn_gym.py -e LunarLander-v2 -n single-hidden
    python dqn_gym.py -e LunarLander-v2 -n two-hidden
    python dqn_gym.py -e LunarLander-v2 -n duelling-dqn

This is how it looks when it's almost solved:

![LunarLander-v2](resources/LunarLander-v2.gif)
