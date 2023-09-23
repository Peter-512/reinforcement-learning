from keras import Model

from agent.episode import Episode
from environment.environment import Environment
from learning.learningstrategy import LearningStrategy


class DeepQLearning(LearningStrategy):
    """
    Two neural nets q1 en q2 are trained together and used to predict the best action.
    These nets are denoted as Q1 and Q2 in the pseudocode.
    This class is INCOMPLETE.
    """
    q1: Model  # keras NN
    q2: Model  # keras NN

    def __init__(self, environment: Environment, batch_size: int, ddqn=False, λ=0.0005, γ=0.99, t_max=200) -> None:
        super().__init__(environment, λ, γ, t_max)
        self.batch_size = batch_size
        self.ddqn = ddqn  # are we using double deep q learning network?
        # TODO: COMPLETE THE CODE

    def on_episode_start(self):
        # TODO: COMPLETE THE CODE
        pass

    def next_action(self, state):
        """ Neural net decides on the next action to take """
        # TODO: COMPLETE THE CODE
        pass

    def learn(self, episode: Episode):
        """ Sample batch from Episode and train NN on sample"""
        # TODO: COMPLETE THE CODE
        super().learn(episode)
        pass

    def build_training_set(self, episode: Episode):
        """ Build training set from episode """
        # TODO: COMPLETE THE CODE
        pass

    def train_network(self, training_set):
        """ Train neural net on training set """
        # TODO: COMPLETE THE CODE
        pass
