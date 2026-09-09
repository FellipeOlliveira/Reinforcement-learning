from typing import NamedTuple

class Params(NamedTuple):
    # variaveis do ambiente
    total_episodes: int # Total de episodios
    map_size: int  # Numero de quadrados do mapa
    seed: int  # Define uma seed padrão para q seja possivel reproduzir os mesmos resultados
    n_runs: int # numero de runs
    proba_frozen: float # Probabilidade de um quadrado esta congelado

    #Variaveis do agente
    learning_rate: float # Learning Rate
    gamma: float # Discount Rate
    epsilon: float # Exploration Propability
    is_slippery: bool # IF true, existe a chance do robo ir para os lados mesmo se escolha ir para frente
    action_size:int # Numero de ações possiveis
    state_size: int # Numero possiveis de estados