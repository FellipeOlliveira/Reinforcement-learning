from typing import NamedTuple

class Params(NamedTuple):
    seed: int
    action_size: int
    state_size: tuple

    learning_rate:float
    epsilon:float
    gamma:float

    natural: bool
    sab: bool
