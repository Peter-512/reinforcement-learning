import numpy as np

from agent.episode import Episode
from agent.percept import Percept
from environment.environment import Environment
from learning.tabular.tabular_learning import TabularLearner


class Qlearning(TabularLearner):

    def __init__(self, environment: Environment, α=0.7, λ=0.0005, γ=0.9, t_max=99, ε_min=0.0005, ε_max=1.0) -> None:
        super().__init__(environment, α, λ, γ, t_max, ε_min, ε_max)

    def learn(self, episode: Episode):
        # implement the Q-learning algorithm
        p: Percept = episode.percepts(1)[-1]
        s = p.state
        a = p.action
        r = p.reward
        t = p.next_state
        self.q_values[s, a] += self.α * (r + self.γ * np.max(self.q_values[t, :]) - self.q_values[s, a])
        super().learn(episode)


class NStepQlearning(TabularLearner):

    def __init__(self, environment: Environment, n: int, α=0.7, λ=0.0005, γ=0.9, t_max=99, ε_min=0.0005,
                 ε_max=1.0) -> None:
        super().__init__(environment, α, λ, γ, t_max, ε_min, ε_max)
        self.n = n  # maximum number of percepts before learning kicks in

    def learn(self, episode: Episode):
        if self.n <= len(episode.percepts(self.n)):
            # implement the N-step Q-learning algorithm
            for percept in episode.percepts(self.n):
                s = percept.state
                a = percept.action
                r = percept.reward
                t = percept.next_state
                self.q_values[s, a] += self.α * (r + self.γ * np.max(self.q_values[t, :]) - self.q_values[s, a])
        super().learn(episode)


class MonteCarloLearning(NStepQlearning):
    def __init__(self, environment: Environment, α=0.7, λ=0.0005, γ=0.9, t_max=99, ε_min=0.0005, ε_max=1.0) -> None:
        super().__init__(environment, np.Inf, α, λ, γ, t_max, ε_min, ε_max)

    def learn(self, episode: Episode):
        self.n = episode.size + 1
        if episode.is_done():
            self.n = episode.size
        super().learn(episode)
