import numpy as np
import matplotlib.pyplot as plt

class Bernoulli_Bandit:
    def __init__(self, n_arms=10, seed=42):
        """
        :param n_arms: Quantidade de Braços
        :param seed: seed para poder replicar o experimento
        """

        self.n_arms = n_arms
        np.random.seed(seed)
        # Gerá a distribuição de probabilidade dos agentes
        self.true_probs = np.random.uniform(0.1, 0.9, n_arms)
        self.best_arm = np.argmax(self.true_probs)
        self.best_prob = self.true_probs[self.best_arm]

    def pull(self, arm):
        """Retorna 1 (sucesso) ou 0 (fracasso) | Simulação de puxar a Alavanca"""
        return 1 if np.random.random() < self.true_probs[arm] else 0


#Teste Rapido para verificar se funcionou
if __name__ == '__main__':
    bandit = Bernoulli_Bandit()

    assert bandit.true_probs[bandit.best_arm] == np.max(bandit.true_probs)

    assert bandit.best_prob == bandit.true_probs[bandit.best_arm]
