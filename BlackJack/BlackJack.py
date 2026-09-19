from functools import total_ordering
from typing import NamedTuple

import numpy as np

import gymnasium as gym

from params import Params
from MonteCarloAgent import MonteCarlo
from EpsilonGreedy import EpsilonGreedy

params = Params(
    n_runs=50
    ,seed=666
    ,action_size=None
    ,state_size=None

    ,learning_rate=0.1
    ,epsilon=0.1
    ,gamma=1

    ,natural=False
    ,sab=False
)

env = gym.make(
    "Blackjack-v1"
    ,render_mode="human"
    ,natural =params.natural #Bota a regra do As(pode ser tanto 1 ou 10)
    ,sab = params.sab #Se vai ter empate ou não(se segue ou n a regra proposta por Surton e Barton)
)

params = params._replace(action_size=env.action_space.n)

params = params._replace(state_size=tuple(space.n for space in env.observation_space))

monte_carlo = MonteCarlo(
    env=env
    ,state_size=params.state_size
    ,action_size=params.action_size
    ,seed=params.seed
    ,learning_rate=params.learning_rate
    ,epsilon=params.epsilon
    ,gamma=params.gamma
)

policy = EpsilonGreedy(
    seed=params.seed
    ,epsilon=params.epsilon
)

def run():


    for i in range(params.n_runs):
        state = env.reset(seed=params.seed)[0]

        terminated = False
        truncated = False
        episode = []

        while not (terminated or truncated):
            action = policy.chose_action(
                env.action_space
                ,state
                ,monte_carlo.qtable)

            print(f"action = {action}")
            print(f"type(action) = {type(action)}")
            print(f"action_space = {env.action_space}")
            print(f"contains = {env.action_space.contains(action)}")

            new_state , reward, terminated, truncated, info = env.step(action=action)
            env.render()

            episode.append((state ,action ,reward))

            state = new_state

            print(f"reward:{reward}\ninfo:{info}\nnew_state:{new_state}")

        monte_carlo.train(qtable=monte_carlo.qtable,episode=episode)
    
        print("End of Episode")
run()