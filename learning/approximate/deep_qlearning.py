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

    def __init__(self, max_size=512):
        self.max_size = max_size
        self.memory = np.empty(max_size, dtype=object)
        self.index = 0
        self.size = 0

    def add(self, experience):
        """ Add experience to memory """
        self.memory[self.index] = experience
        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """ Sample batch from memory """
        indices = np.random.choice(self.size, batch_size, replace=False)
        return self.memory[indices]

    def size(self):
        return self.size


class DeepQLearning(LearningStrategy):
    """
    Two neural nets q1 en q2 are trained together and used to predict the best action.
    These nets are denoted as Q1 and Q2 in the pseudocode.
    This class is INCOMPLETE.
    """
    q1: Model  # keras NN
    q2: Model  # keras NN

    def __init__(self, environment: Environment, batch_size: int, ddqn=False, λ=0.0005, γ=0.99, t_max=200,
                 C=10, max_memory=512, ϵ_min=0.1, ϵ_max=1.0, verbose=False) -> None:
        super().__init__(environment, λ, γ, t_max, ϵ_min, ϵ_max)
        self.batch_size = batch_size
        self.ddqn = ddqn  # are we using double deep q learning network?
        self.q1 = self.build_network()
        self.q2 = self.build_network()
        self.C = C  # how often to update target network
        self.count = 0
        self.replay_memory = ReplayMemory(max_memory)
        self.verbose = 1 if verbose else 0
        print(self.q1.summary())
        print(self.q2.summary())

    def on_episode_start(self):
        super().on_episode_start()
        pass

    def next_action(self, state):
        """ Neural net decides on the next action to take """
        if self.ϵ > random.random():
            return random.randint(0, self.env.n_actions - 1)
        else:
            return np.argmax(self.q1.predict(np.array([state]), verbose=0)[0])

    def learn(self, episode: Episode):
        """ Sample batch from Episode and train NN on sample"""
        self.replay_memory.add(episode.percepts(1)[0])
        if self.replay_memory.size > self.batch_size:
            batch = self.replay_memory.sample(self.batch_size)
            self.learn_from_batch(batch)
        self.decay()
        super().learn(episode)

    def build_training_set(self, batch: [Percept]):
        """ Build training set from episode """
        states = np.array([percept.state for percept in batch])
        next_states = np.array([percept.next_state for percept in batch])
        q_values = self.q1.predict(states, verbose=0, batch_size=self.batch_size)
        q_star_values = self.q2.predict(next_states, verbose=0, batch_size=self.batch_size)
        a_star_values = None
        if self.ddqn:
            a_star_values = self.q1.predict(next_states, verbose=0, batch_size=self.batch_size)

        for i, percept in enumerate(batch):
            if self.ddqn:
                a_star = np.argmax(a_star_values[i])
                q_star = q_star_values[i][a_star]
            else:
                q_star = max(q_star_values[i])
            if percept.done:
                q_values[i][percept.action] = percept.reward
            else:
                q_values[i][percept.action] = percept.reward + self.γ * q_star
        return states, q_values

    def train_network(self, inputs, q_values):
        """ Train neural net on training set """
        self.q1.fit(inputs, q_values, batch_size=self.batch_size, epochs=1, verbose=self.verbose)

    def build_network(self):
        """ Build neural net """
        backend.clear_session()
        backend.set_floatx('float64')
        backend.set_epsilon(1e-4)
        model = models.Sequential()
        model.add(keras.Input(shape=(self.env.state_size,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(self.env.n_actions, activation='linear'))
        model.compile(optimizer="adam", loss='mse', metrics=['accuracy'])
        return model

    def learn_from_batch(self, batch):
        """ Train neural net on batch """
        inputs, q_values = self.build_training_set(batch)
        self.train_network(inputs, q_values)
        self.count += 1
        if self.count % self.C == 0:
            if self.verbose:
                print('updating target network')
            self.q2.set_weights(self.q1.get_weights())
