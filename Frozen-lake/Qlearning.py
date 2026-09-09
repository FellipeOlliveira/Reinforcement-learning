import numpy as np

class Qlearning:
    def __init__(self ,learning_rate ,gamma ,state_size ,action_size):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.state_size = state_size
        self.action_size = action_size
        self.reset_qtable()

    def reset_qtable(self):
        self.qtable = np.zeros((self.state_size,self.action_size))

    def update(self ,state ,action ,reward ,new_state):
        """
        Formula Q learnig para atualizar a recompensa
         Q(s,a) := Q(s,a) + lr [R(s,a) + gamma * max Q(s,a) - Q(s,a)]
        :param state:
        :param action:
        :param reward:
        :param new_state:
        :return:
        """

        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state , action]
        )

        q_update = self.qtable[state , action] + self.learning_rate * delta

        return q_update
