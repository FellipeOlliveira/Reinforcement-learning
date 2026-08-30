import numpy as np
import matplotlib.pyplot as plt

class Agente_Guloso_Epsilon:

    def __init__(self, n_arms, epsilon=0.1, decaying=False, decay_rate=0.999):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decaying = decaying
        self.decay_rate = decay_rate

        # Estimativas do agente
        self.q_values = np.zeros(n_arms)      # Q(a) - É a estimativa atual de cada braço
        self.counts = np.zeros(n_arms)        # N(a) - Quantas vezes cada braço foi escolhido

    def select_action(self):
        """Escolhe uma ação usando ε-greedy"""
        if np.random.random() < self.epsilon:
            # Exploração: escolhe aleatoriamente
            return np.random.randint(self.n_arms)
        else:
            # Exploração(guloso)
            # Caso empate, escolhe aleatoriamente entre as melhores
            max_q = np.max(self.q_values)
            best_arms = np.where(self.q_values == max_q)[0]
            return np.random.choice(best_arms)

    def update(self, arm, reward):
        """Atualiza a estimativa Q do braço escolhido (média incremental)"""
        self.counts[arm] += 1
        # Fórmula incremental da média: Q_novo = Q_antigo + (1/N) * (R - Q_antigo)
        self.q_values[arm] += (1 / self.counts[arm]) * (reward - self.q_values[arm])
        #OBS: ultilizei essa formula para não precisar armazenar todas as estimativas antigas do agente

        # Decai o epsilon se estiver no modo decaying
        if self.decaying:
            self.epsilon = max(0.01, self.epsilon * self.decay_rate)  # nunca vai abaixo de 1%

if __name__ == "__main__":
    #Teste para saber se esta funcionando

    agente = Agente_Guloso_Epsilon(n_arms=4,epsilon=0.1)

    # Q-values
    assert len(agente.q_values) == 4
    assert np.all(agente.q_values == 0)

    # Contadores
    assert len(agente.counts) == 4
    assert np.all(agente.counts == 0)

    # Update
    agente.update(2, 1)

    assert agente.counts[2] == 1
    assert agente.q_values[2] == 1

    # Média incremental
    agente.update(2, 0)

    assert agente.counts[2] == 2
    assert agente.q_values[2] == 0.5

    # Caso de braços empatados
    agente.q_values = np.array([
        0.5,
        0.2,
        0.5,
        0.1
    ])

    max_q = np.max(agente.q_values)
    best_arms = np.where(agente.q_values == max_q)[0]

    assert np.array_equal(best_arms, np.array([0, 2]))

    # Selecionando ação/gulosa
    agente.epsilon = 0

    for _ in range(100):
        action = agente.select_action()

        assert action in [0, 2]


    # Epsilon decaying
    agente_decay = Agente_Guloso_Epsilon(
        n_arms=4,
        epsilon=0.1,
        decaying=True,
        decay_rate=0.9
    )

    agente_decay.update(0, 1)

    assert np.isclose(agente_decay.epsilon, 0.09)
