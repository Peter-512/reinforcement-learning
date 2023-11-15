import random

import keras
import numpy as np
from keras import Model, layers, models, backend

from agent.episode import Episode
from agent.percept import Percept
from environment.environment import Environment
from learning.learningstrategy import LearningStrategy


class ReplayMemory:
    """ Replay memory for experience replay """

    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.memory = []

    def add(self, experience):
        """ Add experience to memory """
        self.memory.append(experience)
        # if len(self.memory) > self.max_size:
        #     self.memory.pop(0)

    def sample(self, batch_size):
        """ Sample batch from memory """
        return random.sample(self.memory, batch_size)

    def size(self):
        return len(self.memory)


class DeepQLearning(LearningStrategy):
    """
    Two neural nets q1 en q2 are trained together and used to predict the best action.
    These nets are denoted as Q1 and Q2 in the pseudocode.
    This class is INCOMPLETE.
    """
    q1: Model  # keras NN
    q2: Model  # keras NN

    def __init__(self, environment: Environment, batch_size: int, ddqn=False, λ=0.0005, γ=0.99, t_max=200,
                 C=10, max_memory=10_000) -> None:
        super().__init__(environment, λ, γ, t_max)
        self.batch_size = batch_size
        self.ddqn = ddqn  # are we using double deep q learning network?
        self.q1 = self.build_network()
        self.q2 = self.build_network()
        self.C = C  # how often to update target network
        self.count = 0
        self.replay_memory = ReplayMemory(max_memory)
        print(self.q1.summary())
        print(self.q2.summary())

    def on_episode_start(self):
        # TODO: COMPLETE THE CODE
        pass

    def next_action(self, state):
        """ Neural net decides on the next action to take """
        if self.ϵ > random.random():
            return random.randint(0, self.env.n_actions - 1)
        else:
            return np.argmax(self.q1.predict(state))

    def learn(self, episode: Episode):
        """ Sample batch from Episode and train NN on sample"""
        self.replay_memory.add(episode)
        if self.replay_memory.size() > self.batch_size:
            batch = self.replay_memory.sample(self.batch_size)
            self.learn_from_batch(batch)
        super().learn(episode)

    def build_training_set(self, batch: [Percept]):
        """ Build training set from episode """
        training_set = []
        for percept in batch:
            q = self.q1.predict(percept.state)
            if self.ddqn:
                a_star = np.argmax(self.q1(percept.next_state))
                q_star = self.q2(percept.next_state)[a_star]
            else:
                q_star = max(self.q2(percept.next_state))
            if percept.done:
                q[percept.action] = percept.reward
            else:
                q[percept.action] = percept.reward + self.γ * q_star
            training_set.append((percept.state, q(percept.state)))
        return training_set

    def train_network(self, training_set):
        """ Train neural net on training set """
        for state, q in training_set:
            self.q1.fit(state, q)

    def build_network(self):
        """ Build neural net """
        backend.clear_session()
        backend.set_floatx('float64')
        backend.set_epsilon(1e-4)
        return models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.env.state_size,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.env.n_actions, activation='linear')
        ])

    def learn_from_batch(self, batch):
        """ Train neural net on batch """
        training_set = self.build_training_set(batch)
        self.train_network(training_set)
        if self.count % self.C == 0:
            self.q2.set_weights(self.q1.get_weights())
