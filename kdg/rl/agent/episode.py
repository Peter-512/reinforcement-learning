from collections import deque

from be.kdg.rl.agent.percept import Percept
from be.kdg.rl.environment.environment import Environment


class Episode:
    """
    A collection of Percepts forms an Episode. A Percept is added per step/time t.
    The Percept contains the state, action, reward and next_state.
    This class is INCOMPLETE
    """

    def __init__(self, env: Environment) -> None:
        self._env = env
        self._percepts: [Percept] = deque()

    def add(self, percept: Percept):
        self._percepts.append(percept)

    def percepts(self, n: int):
        """ Return n final percepts from this Episode """
        # TODO: COMPLETE THE CODE
        pass

    def compute_returns(self) -> None:
        """ For EACH Percept in the Episode, calculate its discounted return Gt"""
        # TODO: COMPLETE THE CODE
        pass

    def sample(self, batch_size: int):
        """ Sample and return a random batch of Percepts from this Episode """
        # TODO: COMPLETE THE CODE
        pass

    @property
    def size(self):
        return len(self._percepts)
