from abc import abstractmethod
import numpy as np

from agent.episode import Episode
from agent.percept import Percept
from environment.environment import Environment
from learning.learningstrategy import LearningStrategy
from learning.tabular.tabular_learning import TabularLearner
from stats.reward_signal import RewardSignal, WinPercentage
from stats.graph_plotter import Stats
from environment.markovdecisionprocess import MarkovDecisionProcess


class Agent:

    def __init__(self, environment: Environment, learning_strategy: LearningStrategy, n_episodes=10_000,
                 stats_episodes=200):
        super().__init__()
        self.env = environment
        self.learning_strategy = learning_strategy
        self.episodes: [Episode] = []
        self.n_episodes = n_episodes  # total episodes
        self.episode_count = 0
        self.stats_episodes = stats_episodes
        self.stats_generator = Stats(environment.map_shape, stats_episodes)
        self.reward_signal = RewardSignal(learning_strategy.__class__.__name__)
        self.win_percentage = WinPercentage(learning_strategy.__class__.__name__)

    @abstractmethod
    def train(self) -> None:
        pass

    @property
    def done(self):
        return self.episode_count > self.n_episodes


class SuperAgent(Agent):
    def __init__(self, mdp: MarkovDecisionProcess, n_episodes=10_000):
        self.mdp = mdp
        self.env = mdp.env
        super().__init__(mdp.env, n_episodes=n_episodes, learning_strategy=None)

    def train(self):
        while not self.done:
            # start a new episode
            episode = Episode(self.env)
            self.episodes.append(episode)
            # initialize the start state
            state, _ = self.env.reset()

            while True:
                action = self.env.action_space.sample()
                t, r, terminated, truncated, info = self.env.step(action)
                print('state:', state, 'action:', action, 'reward:', r, 'next_state:', t, 'terminated:', terminated)
                if terminated and r == 0:
                    r = -1
                elif not terminated:
                    r = -0.01
                input()
                self.env.render()
                percept = Percept((state, action, r, t, terminated))
                episode.add(percept)
                self.mdp.update((state, action, r, t, terminated))
                state = percept.next_state
                if percept.done:
                    break

            self.episode_count += 1
            self.win_percentage.add(r)

        self.env.close()
        self.win_percentage.plot()
        print(self.mdp.P)


class TabularAgent(Agent):

    def __init__(self, environment: Environment, learning_strategy: TabularLearner, n_episodes=10_000) -> None:
        super().__init__(environment, learning_strategy, n_episodes)

    def train(self) -> None:
        super(TabularAgent, self).train()

        # as longs as the agents hasn't reached the maximum number of episodes
        while not self.done:

            # start a new episode
            episode = Episode(self.env)
            self.episodes.append(episode)
            # initialize the start state
            state, _ = self.env.reset()
            # reset the learning strategy for the new episode
            self.learning_strategy.on_episode_start()

            # while the episode isn't finished by length
            while not self.learning_strategy.done():

                # learning strategy (policy) determines next action to take
                action = self.learning_strategy.next_action(state)
                # agent observes the results of his action
                # step method returns a tuple with values (s', r, terminated, truncated, info)
                t, r, terminated, truncated, info = self.env.step(action)

                # render environment (don't render every step, only every X-th, or at the end of the learning process)
                self.env.render()

                # create Percept object from observed values state,action,r,s' (SARS') and terminate flag, but
                # ignore values truncated and info
                percept = Percept((state, action, r, t, terminated))
                # print(percept)

                # add the newly created Percept to the Episode
                episode.add(percept)

                # learn from Percepts in Episode
                self.learning_strategy.learn(episode)

                # update Agent's state
                state = percept.next_state

                # break if episode is over and inform learning strategy
                if percept.done:
                    self.learning_strategy.on_episode_end()
                    self.reward_signal.add(r)
                    self.win_percentage.add(r)
                    break

            # end episode
            self.episode_count += 1

            if self.episode_count % self.stats_episodes == 0:
                self.stats_generator.generate_quiver_from_data(
                    np.reshape(self.learning_strategy.get_ideal_path(), self.env.map_shape),
                    self.env.map_arr)
            if self.episode_count % 500 == 0:
                # self.reward_signal.plot()
                self.win_percentage.plot()

        self.env.close()
        self.reward_signal.plot()
        self.win_percentage.plot()
        self.stats_generator.generate_final_policy_gif(
            np.array(self.learning_strategy.get_ideal_path()).reshape(self.env.map_shape),
            self.env.map_arr, self.learning_strategy.__class__.__name__)
        print('q table:\n', self.learning_strategy.q_values)
        print('v table:\n', self.learning_strategy.v_values.reshape(self.env.map_shape))
        print('π table:\n',
              self.learning_strategy.π.reshape(self.env.n_actions, self.env.map_shape[0], self.env.map_shape[1]))
