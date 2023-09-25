from abc import ABC

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from environment.environment import Environment


class OpenAIGym(Environment, ABC):
    """
    Superclass for all kinds of OpenAI environments
    Wrapper for all OpenAI Environments
    """

    def __init__(self, name: str, render=False) -> None:
        super().__init__()
        self._name = name
        render_mode = 'human' if render else None
        self._env: TimeLimit = TimeLimit(gym.make(name, render_mode=render_mode), max_episode_steps=100)

    def reset(self):
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def render(self):
        if self._env.render_mode:
            self._env.render()

    def close(self) -> None:
        self._env.close()

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def n_actions(self):
        return self._env.action_space.n

    @property
    def state_size(self):
        if self.isdiscrete:
            return self._env.observation_space.n
        else:
            return self._env.observation_space.shape[0]

    @property
    def isdiscrete(self) -> bool:
        return hasattr(self._env.observation_space, 'n')

    @property
    def name(self) -> str:
        return self._name


class FrozenLakeEnvironment(OpenAIGym):

    def __init__(self, render=False) -> None:
        super().__init__(name="FrozenLake-v1", render=render)


class CartPoleEnvironment(OpenAIGym):

    def __init__(self, render=False) -> None:
        super().__init__(name="CartPole-v0", render=render)
