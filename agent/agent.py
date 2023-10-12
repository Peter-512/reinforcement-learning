from abc import abstractmethod

from agent.episode import Episode
from agent.percept import Percept
from environment.environment import Environment
from learning.learningstrategy import LearningStrategy
from learning.tabular.tabular_learning import TabularLearner


class Agent:

    def __init__(self, environment: Environment, learning_strategy: LearningStrategy, n_episodes=10000):
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


class MarkovAgent(Agent):
    def __int__(self, environment: Environment, n_episodes=10_000) -> None:
        super().__init__(environment, None, n_episodes)


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

                # update Agent's state
                state = percept.next_state

                # learn from Percepts in Episode
                self.learning_strategy.learn(episode)

                # update state
                state = percept.next_state

                # break if episode is over and inform learning strategy
                if percept.done:
                    self.learning_strategy.on_episode_end()
                    break

            # end episode
            self.episode_count += 1

        self.env.close()
