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

    def __init__(self, environment: Environment, α=0.7, λ=0.0005, γ=0.9, t_max=99, episode_stats=200,
                 ε_max=1.0) -> None:
        super().__init__(environment, λ, γ, t_max, episode_stats, ε_max)
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
        if self.τ % self.stats_generator.e == 0 and self.t == 0:
            self.stats_generator.generate_quiver_from_data(np.reshape(self.get_ideal_path(), self.env.map_shape),
                                                           self.env.map_arr)
        self.evaluate()
        self.improve()
        super().learn(episode)

    def on_episode_start(self):
        self.t = 0

    def next_action(self, s: int):
        # if self.ε > np.random.random():
        #     return self.env.action_space.sample()
        return np.random.choice(np.arange(self.env.n_actions), p=self.π[:, s])

    def evaluate(self):
        for s in range(self.env.state_size):
            self.v_values[s] = np.max(self.q_values[s])

    def improve(self):
        for s in range(self.env.state_size):
            a_star = np.random.choice(np.flatnonzero(self.q_values[s] == self.q_values[s].max()))

            for a in range(self.env.n_actions):
                if a == a_star:
                    self.π[a, s] = 1 - self.ε + self.ε / self.env.n_actions
                else:
                    self.π[a, s] = self.ε / self.env.n_actions
        self.decay()

    def __repr__(self):
        return f"TabularLearner(α: {self.α}, λ: {self.λ}, γ: {self.γ}, t_max: {self.t_max})"
