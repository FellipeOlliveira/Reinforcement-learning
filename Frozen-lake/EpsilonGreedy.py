import numpy as np

class EpsilonGreedy:
    def __init__(self, epsilon,seed=None):
        self.epsilon = epsilon
        self.rng = seed

    def chose_action(self ,action_space ,state ,qtable):
        """
        Escolhe uma ação `a` no estado atual(s)
        :param action_space:
        :param state:
        :param qtable:
        :return:
        """

        #Aleatoriza a ação
        explor_exploit_tradeoff = self.rng.uniform(0,1)

        #Exploration
        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample()

        #Exploitation (Pega o melhor Q-value para esse estado
        else:
            max_ids = np.where(qtable[state, :] == max(qtable[state, :]))[0]
            action = self.rng.choice(max_ids)

        return action
