from functools import total_ordering
from typing import NamedTuple

import numpy as np

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from params import Params
from Qlearning import Qlearning
from EpsilonGreedy import EpsilonGreedy


params = Params(
    total_episodes=500
    ,learning_rate=0.8
    ,gamma=0.95
    ,epsilon=0.1
    ,map_size=5
    ,seed=666
    ,is_slippery=False
    ,n_runs=2
    ,action_size=None
    ,state_size=None
    ,proba_frozen=0.85
)

env = gym.make(
    "FrozenLake-v1"
    ,is_slippery=params.is_slippery
    ,render_mode="human"
    ,desc=generate_random_map(
        size=params.map_size
        ,p=params.proba_frozen
        ,seed=params.seed
    )
)

params = params._replace(action_size=env.action_space.n)
params = params._replace(state_size=env.observation_space.n)
rng = np.random.default_rng(params.seed)

qlearning = Qlearning(
    learning_rate=params.learning_rate
    ,gamma=params.gamma
    ,state_size=params.state_size
    ,action_size=params.action_size
)

policy = EpsilonGreedy(
    epsilon=params.epsilon
    ,seed=rng
)

def run():
    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes,params.n_runs))
    episodes = np.arange((params.total_episodes))
    qtables = np.zeros((params.n_runs,params.state_size,params.action_size))
    all_states = []
    all_actions = []

    for run in range(params.n_runs):
        qlearning.reset_qtable()

        for episode in episodes:
            state = env.reset(seed=params.seed)[0]
            step = 0
            done = False
            total_rewards = 0

            while not done:
                action = policy.chose_action(
                    action_space=env.action_space
                    ,state=state
                    ,qtable=qlearning.qtable
                )

                                # Log das ações e estados
                all_states.append(state)
                all_actions.append(action)

                # toma a decisão (a) e observa a recompensa (r) do estado (s')
                new_state ,reward ,terminated ,truncated ,info = env.step(action)

                env.render()

                done = terminated or truncated

                qlearning.qtable[state ,action]  = qlearning.update(
                    state, action ,reward ,new_state
                )

                total_rewards += reward

                step += 1

                state = new_state

            rewards[episode, run] = total_rewards
            steps[episode, run] = step
        qtables[run ,: ,:] = qlearning.qtable


    return rewards, steps , episodes, qtables, all_states, all_actions


run()