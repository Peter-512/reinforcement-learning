from abc import abstractmethod
import numpy as np

from agent.episode import Episode
from agent.percept import Percept
from environment.environment import Environment
from learning.learningstrategy import LearningStrategy
from learning.approximate.deep_qlearning import DeepQLearning
from learning.tabular.tabular_learning import TabularLearner
from stats.reward_signal import RewardSignal, WinPercentage, EpsilonDecay, EpisodeLength
from stats.graph_plotter import Stats
from stats.savepoint import SavePoint


class Agent:

    def __init__(self, environment: Environment, learning_strategy: LearningStrategy, n_episodes=10_000):
        super().__init__()
        self.env = environment
        self.learning_strategy = learning_strategy
        self.episodes: [Episode] = []
        self.n_episodes = n_episodes  # total episodes
        self.episode_count = 0

    @abstractmethod
    def train(self) -> None:
        pass

    @property
    def done(self):
        return self.episode_count > self.n_episodes


class ApproximateAgent(Agent):
    def __init__(self, environment: Environment, learning_strategy: LearningStrategy, n_episodes=10_000,
                 savepoint_base_path='savepoints') -> None:
        super().__init__(environment, learning_strategy, n_episodes)
        self.episode_duration_plotter = EpisodeLength(learning_strategy.__class__.__name__)
        self.savepoint = SavePoint(learning_strategy.__class__.__name__, savepoint_base_path)

    def train(self) -> None:
        super().train()

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
                done = True if self.learning_strategy.t >= 500 or terminated else False
                percept = Percept((state, action, r, t, terminated))
                # print(percept)

                # add the newly created Percept to the Episode
                episode.add(percept)

                # learn from Percepts in Episode
                self.learning_strategy.learn(episode)

                # update Agent's state
                state = percept.next_state

                if done:
                    self.learning_strategy.on_episode_end()
                    self.episode_duration_plotter.add(self.learning_strategy.t)
                    break

            self.episode_count += 1
            print('episode:', self.episode_count)

            if self.episode_count % 50 == 0:
                self.episode_duration_plotter.plot()
                self.episode_duration_plotter.reset()

            if self.episode_count % 100 == 0:
                self.savepoint.save(self)

        self.env.close()


class TabularAgent(Agent):

    def __init__(self, environment: Environment, learning_strategy: TabularLearner, n_episodes=10_000,
                 stats_episodes=200) -> None:
        super().__init__(environment, learning_strategy, n_episodes)
        self.stats_episodes = stats_episodes
        self.stats_generator = Stats(environment.map_shape, stats_episodes)
        self.reward_signal = RewardSignal(learning_strategy.__class__.__name__)
        self.win_percentage = WinPercentage(learning_strategy.__class__.__name__)
        self.epsilon_decay = EpsilonDecay(learning_strategy.__class__.__name__)
        self.episode_duration_plotter = EpisodeLength(learning_strategy.__class__.__name__)

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
                    self.epsilon_decay.add(self.learning_strategy.ε)
                    self.episode_duration_plotter.add(self.learning_strategy.t)
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
                # self.epsilon_decay.plot()
                # self.episode_duration_plotter.plot()

        self.env.close()
        self.reward_signal.plot()
        self.win_percentage.plot()
        self.epsilon_decay.plot()
        self.episode_duration_plotter.plot()
        self.stats_generator.generate_final_policy_gif(
            np.array(self.learning_strategy.get_ideal_path()).reshape(self.env.map_shape),
            self.env.map_arr, self.learning_strategy.__class__.__name__)
        print('q table:\n', self.learning_strategy.q_values)
        print('v table:\n', self.learning_strategy.v_values.reshape(self.env.map_shape))
        print('π table:\n',
              self.learning_strategy.π.reshape(self.env.n_actions, self.env.map_shape[0], self.env.map_shape[1]))
