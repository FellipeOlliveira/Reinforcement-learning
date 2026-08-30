import numpy as np
import matplotlib.pyplot as plt

from agent import Agente_Guloso_Epsilon
from enviroment import Bernoulli_Bandit

def run_bandit(agent, bandit, n_steps=2000):
    rewards = np.zeros(n_steps)
    regrets = np.zeros(n_steps)
    cumulative_regret = 0

    for t in range(n_steps):
        arm = agent.select_action()
        reward = bandit.pull(arm)
        agent.update(arm, reward)

        rewards[t] = reward
        # Regret = recompensa do melhor braço - recompensa obtida
        cumulative_regret += (bandit.best_prob - bandit.true_probs[arm])
        regrets[t] = cumulative_regret

    return rewards, regrets


def main():
    n_arms = 10
    n_steps = 2000
    n_runs = 50          # quantas vezes vamos repetir o experimento (suaviza a curva no gráfico final)

    # Três agentes que será usado de teste
    agents_config = {
        "Greedy (ε=0)":          {"epsilon": 0.0,  "decaying": False},
        "ε-greedy (ε=0.1)":      {"epsilon": 0.1,  "decaying": False},
        "ε-greedy decrescente":  {"epsilon": 1.0,  "decaying": True, "decay_rate": 0.995},
    }

    results = {}

    for name, config in agents_config.items():
        all_rewards = []
        all_regrets = []

        for run in range(n_runs):
            bandit = Bernoulli_Bandit(n_arms=n_arms, seed=run)  # seed diferente a cada run
            agent = Agente_Guloso_Epsilon(n_arms=n_arms, **config)
            rewards, regrets = run_bandit(agent, bandit, n_steps)
            all_rewards.append(rewards)
            all_regrets.append(regrets)

        # Média entre as várias runs
        results[name] = {
            "avg_reward": np.mean(all_rewards, axis=0),
            "avg_regret": np.mean(all_regrets, axis=0)
        }

    #VISUALIZAÇÃO
    plt.figure(figsize=(14, 5))

    # Gráfico 1: Recompensa média
    plt.subplot(1, 2, 1)
    for name, data in results.items():
        # Média móvel para deixar a curva mais legível
        smoothed = np.convolve(data["avg_reward"], np.ones(50)/50, mode='valid')
        plt.plot(smoothed, label=name)
    plt.xlabel("Passos")
    plt.ylabel("Recompensa média")
    plt.title("Recompensa média ao longo do tempo")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Gráfico 2: Regret acumulado
    plt.subplot(1, 2, 2)
    for name, data in results.items():
        plt.plot(data["avg_regret"], label=name)
    plt.xlabel("Passos")
    plt.ylabel("Regret acumulado")
    plt.title("Regret acumulado (quanto deixamos de ganhar)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Mostra as probabilidades verdadeiras para comparar os resultados
    print("\nProbabilidades verdadeiras dos braços:")
    bandit_demo = Bernoulli_Bandit(n_arms=n_arms, seed=42)
    for i, p in enumerate(bandit_demo.true_probs):
        marker = " ← MELHOR" if i == bandit_demo.best_arm else ""
        print(f"Braço {i}: {p:.3f}{marker}")

if __name__ == '__main__':
    main()