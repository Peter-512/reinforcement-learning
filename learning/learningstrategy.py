from abc import ABC, abstractmethod

import numpy as np

from agent.episode import Episode
from environment.environment import Environment
from stats.graph_plotter import Stats


class LearningStrategy(ABC):
    """
    Implementations of this class represent a Learning Method
    This class is INCOMPLETE
    """
    env: Environment

    def __init__(self, environment: Environment, λ, γ, t_max, episode_stats=200) -> None:
        self.env = environment
        self.λ = λ  # exponential decay rate used for exploration/exploitation (given)
        self.γ = γ  # discount rate for exploration (given)
        self.ε_max = 1.0  # Exploration probability at start (given)
        self.ε_min = 0.0005  # Minimum exploration probability (given)

        self.ε = self.ε_max  # (decaying) probability of selecting random action according to ε-soft policy
        self.t_max = t_max  # upper limit for episode
        self.t = 0  # episode time step
        self.τ = 0  # overall time step
        self.stats_generator = Stats(self.env.map_shape, episode_stats)

    @abstractmethod
    def next_action(self, state):
        pass

    @abstractmethod
    def learn(self, episode: Episode):
        # at this point subclasses insert their implementation
        # see for example be\kdg\rl\learning\tabular\tabular_learning.py
        self.t += 1

    @abstractmethod
    def on_episode_start(self):
        """
        Implements all necessary initialization that needs to be done at the start of new Episode
        Each subclass learning algorithm should decide what to do here
        """
        pass

    def done(self):
        return self.t > self.t_max

    def decay(self):
        # Reduce epsilon ε, because we need less and less exploration as time progresses
        self.ε = self.ε_min + (self.ε_max - self.ε_min) * np.exp(-self.λ * self.τ)

    def on_episode_end(self):
        self.τ += 1

    def show_policy(self):
        print("Policy:")
        ideal_path = self.get_ideal_path()
        self.stats_generator.generate_final_policy_gif(np.array(ideal_path).reshape(self.env.map_shape),
                                                       self.env.map_arr)

        direction = {0: '⬅', 1: '⬇', 2: '➡︎', 3: '⬆'}
        for i in range(len(ideal_path)):
            ideal_path[i] = direction[ideal_path[i]]

        for i in range(len(ideal_path)):
            # flatten the map
            map = np.array(self.env.map_arr).flatten()
            if map[i] == b'H':
                ideal_path[i] = '🚫'
            elif map[i] == b'G':
                ideal_path[i] = '🏁'
        print(np.array(ideal_path).reshape(self.env.map_shape))

    def get_ideal_path(self):
        pi = self.π
        ideal_path = []
        # loop through the nested array pi breath first and find the largest value for each state
        for i in range(len(pi[0])):
            one = pi[0][i]
            two = pi[1][i]
            three = pi[2][i]
            four = pi[3][i]
            l = [one, two, three, four]
            ideal_path.append(np.argmax(l))
        return ideal_path
