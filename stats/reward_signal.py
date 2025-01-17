from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

from itertools import count

import numpy as np

from stats.graph_plotter import create_directory_if_not_exists, clear_directory


class Plotter(ABC):
    def __init__(self, strategy):
        self.strategy = strategy
        self.episodes = []
        self.episodes_counter = count()

    @abstractmethod
    def add(self, value):
        pass

    @abstractmethod
    def plot(self):
        pass


class RewardSignal(Plotter):
    def __init__(self, strategy):
        super().__init__(strategy)
        self.cum_rewards = []

    def add(self, value):
        self.cum_rewards.append(value + self.cum_rewards[-1] if len(self.cum_rewards) > 0 else 0)
        self.episodes.append(next(self.episodes_counter))

    def plot(self):
        plt.plot(self.episodes, self.cum_rewards, linewidth=2.0)
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Rewards for ' + self.strategy + ' Strategy')
        plt.grid(True)
        plt.show()


class WinPercentage(Plotter):
    def __init__(self, strategy):
        super().__init__(strategy)
        self.win_percentages = []
        self.cum_rewards = 0

    def add(self, value):
        self.cum_rewards += value
        self.episodes.append(next(self.episodes_counter))
        self.win_percentages.append((self.cum_rewards / len(self.episodes)) * 100)

    def plot(self):
        plt.plot(self.episodes, self.win_percentages, linewidth=2.0)
        plt.xlabel('Episodes')
        plt.ylabel('Win Percentage')
        plt.title('Win Percentage for ' + self.strategy + ' Strategy')
        plt.ylim(0, 100)
        plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        plt.grid(True)
        plt.show()


class EpsilonDecay(Plotter):
    def __init__(self, strategy):
        super().__init__(strategy)
        self.epsilon_values = []

    def add(self, value):
        self.episodes.append(next(self.episodes_counter))
        self.epsilon_values.append(value)

    def plot(self):
        plt.plot(self.episodes, self.epsilon_values, linewidth=2.0)
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon Value')
        plt.title('Epsilon values for ' + self.strategy + ' Strategy')
        plt.ylim(0, 1)
        plt.yticks([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
        plt.grid(True)
        plt.show()


class EpisodeLength(Plotter):
    def __init__(self, strategy):
        super().__init__(strategy)
        create_directory_if_not_exists("./screenshots")
        clear_directory("./screenshots")
        self.episode_lengths = []
        self.all_episode_lengths = []
        self.averages = []

    def add(self, value):
        self.episodes.append(next(self.episodes_counter))
        self.episode_lengths.append(value)
        self.all_episode_lengths.append(value)
        self.averages.append(np.mean(self.all_episode_lengths))

    def reset(self):
        self.episodes = []
        self.episode_lengths = []

    def moving_average(self, window):
        return self.averages[-window:]

    def plot(self):
        plt.plot(self.episodes, self.episode_lengths, linewidth=2.0, label='Episode Length')
        moving_averages = self.moving_average(len(self.episodes))
        plt.plot(self.episodes, moving_averages, linewidth=2.0, label='Moving Average')
        plt.ylim(0, 505)
        plt.grid()
        plt.hlines(195, self.episodes[0], self.episodes[-1], colors='red', linestyles='dashed', label='Goal')
        plt.legend()
        plt.xlabel('Episodes')
        plt.ylabel('Episode Length')
        plt.title('Episode Length for ' + self.strategy + ' Strategy')
        plt.savefig(f"./screenshots/episodes-{self.episodes[0]}-{self.episodes[-1]}.png")
        plt.show()
