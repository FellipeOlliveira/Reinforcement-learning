import numpy as np

class MonteCarlo:
    def __init__(self,env,state_size,action_size,seed,learning_rate,epsilon,gamma):
        self.env = env
        self.action_size = action_size
        self.state_size = state_size
        self.seed = seed

        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma

        self.reset_qtable()

    def reset_qtable(self):
        self.qtable = np.zeros(
            self.state_size + (self.action_size,)
            )

    def train(self, qtable,episode):
        G = 0

        for state ,action ,reward in reversed(episode):
            G = reward + self.gamma * G

            qtable[state,action] = (
                qtable[state,action] + self.learning_rate * (G - qtable[state,action])
            )

        return qtable