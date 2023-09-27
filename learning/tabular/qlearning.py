import numpy as np

from agent.episode import Episode
from agent.percept import Percept
from environment.environment import Environment
from learning.tabular.tabular_learning import TabularLearner


class Qlearning(TabularLearner):

    def __init__(self, environment: Environment, α=0.7, λ=0.0005, γ=0.9, t_max=99) -> None:
        TabularLearner.__init__(self, environment, α, λ, γ, t_max)

    def learn(self, episode: Episode):
        # implement the Q-learning algorithm
        p: Percept = episode.percepts(1)
        s = p.state
        a = p.action
        r = p.reward
        self.q_values[s, a] = self.q_values[s, a] + self.α * (
                r + self.γ * np.max(self.q_values[p.next_state, :]) - self.q_values[s, a])

        super().learn(episode)


class NStepQlearning(TabularLearner):

    def __init__(self, environment: Environment, n: int, α=0.7, λ=0.0005, γ=0.9, t_max=99) -> None:
        TabularLearner.__init__(self, environment, α, λ, γ, t_max)
        self.n = n  # maximum number of percepts before learning kicks in
        self.percepts = []  # this will buffer the percepts

    def learn(self, episode: Episode):
        if self.n > len(episode.percepts(self.n)):
            return
        # implement the N-step Q-learning algorithm
        for percept in episode.percepts(self.n):
            s = percept.state
            a = percept.action
            r = percept.reward
            self.q_values[s, a] = self.q_values[s, a] + self.α * (
                    r + self.γ * np.max(self.q_values[percept.next_state, :]) - self.q_values[s, a])
        super().learn(episode)


class MonteCarloLearning(TabularLearner):
    def __init__(self, environment: Environment, n: int, α=0.7, λ=0.0005, γ=0.9, t_max=99) -> None:
        TabularLearner.__init__(self, environment, α, λ, γ, t_max)
        self.n = n  # maximum number of percepts before learning kicks in
        self.percepts = []  # this will buffer the percepts

    def learn(self, episode: Episode):
        if not episode.is_done():
            return
        for percept in episode.percepts(self.n):
            s = percept.state
            a = percept.action
            r = percept.reward
            self.q_values[s, a] = self.q_values[s, a] + self.α * (
                    r + self.γ * np.max(self.q_values[percept.next_state, :]) - self.q_values[s, a])
        super().learn(episode)