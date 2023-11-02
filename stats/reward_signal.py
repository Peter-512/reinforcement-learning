from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

from itertools import count


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
