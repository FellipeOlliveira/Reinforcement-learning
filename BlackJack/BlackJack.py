import gymnasium as gym

from params import Params
from MonteCarloAgent import MonteCarlo
from EpsilonGreedy import EpsilonGreedy

params = Params(
    n_runs=500
    ,seed=666
    ,action_size=None
    ,state_size=None

    ,learning_rate=0.1
    ,epsilon=0.1
    ,gamma=1

    ,natural=True
    ,sab=True
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
        state = env.reset()[0]

        terminated = False
        truncated = False
        episode = []

        while not (terminated or truncated):
            env.render()
            action = policy.chose_action(
                env.action_space
                ,state
                ,monte_carlo.qtable)

            new_state , reward, terminated, truncated, info = env.step(action=action)
            

            episode.append((state ,action ,reward))

            state = new_state

        monte_carlo.train(qtable=monte_carlo.qtable,episode=episode)
    
        print("End of Episode {i}")
run()