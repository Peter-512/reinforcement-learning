from abc import abstractmethod

import numpy as np
from numpy import ndarray

from agent.episode import Episode
from environment.environment import Environment
from learning.learningstrategy import LearningStrategy


class TabularLearner(LearningStrategy):
    """
    A tabular learner implements a tabular method such as Q-Learning, N-step Q-Learning, ...
    """
    π: ndarray
    v_values: ndarray
    q_values: ndarray

    def __init__(self, environment: Environment, α=0.7, λ=0.0005, γ=0.9, t_max=99) -> None:
        super().__init__(environment, λ, γ, t_max)
        # learning rate
        self.α = α

        # policy table
        self.π = np.full((self.env.n_actions, self.env.state_size), fill_value=1 / self.env.n_actions)

        # state value table
        self.v_values = np.zeros((self.env.state_size,))

        # state-action table
        self.q_values = np.zeros((self.env.state_size, self.env.n_actions))

    @abstractmethod
    def learn(self, episode: Episode):
        # subclasses insert their implementation at this point
        # see for example be\kdg\rl\learning\tabular\qlearning.py
        self.evaluate()
        self.improve()
        super().learn(episode)

    def on_episode_start(self):
        self.t = 0

    def next_action(self, s: int):
        # TODO: COMPLETE THE CODE, REPLACE THE LINE BELOW
        return self.env.action_space.sample()

    def evaluate(self):
        # TODO: COMPLETE THE CODE
        pass

    def improve(self):
        # TODO: COMPLETE THE CODE
        pass
