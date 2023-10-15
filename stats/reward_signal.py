import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import count


class RewardSignal:
    def __init__(self, strategy):
        self.strategy = strategy
        self.cum_rewards = []
        self.episodes = []
        self.episodes_counter = count()
        self.ani = None

    def animate(self, i):
        plt.cla()
        plt.plot(self.episodes, self.cum_rewards, linewidth=2.0)

    def add(self, value):
        self.cum_rewards.append(value + self.cum_rewards[-1] if len(self.cum_rewards) > 0 else 0)
        self.episodes.append(next(self.episodes_counter))

    def setup(self):
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Rewards for ' + self.strategy + ' Strategy')
        plt.grid(True)
        # self.ani = FuncAnimation(plt.gcf(), self.animate, interval=1000, blit=False, cache_frame_data=False)

    def plot(self):
        plt.plot(self.episodes, self.cum_rewards, linewidth=2.0)
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Rewards for ' + self.strategy + ' Strategy')
        plt.grid(True)
        plt.show()
