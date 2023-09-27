import numpy as np

from agent.agent import TabularAgent, Agent
from environment.openai import FrozenLakeEnvironment
from learning.tabular.qlearning import Qlearning, NStepQlearning, MonteCarloLearning

if __name__ == '__main__':
    # example use of the code base
    environment = FrozenLakeEnvironment(render=False)

    # create an Agent that uses Qlearning Strategy
    # agent: Agent = TabularAgent(environment, Qlearning(environment))
    # agent.train()

    # create an Agent that uses NStepQlearning Strategy
    agent: Agent = TabularAgent(environment, MonteCarloLearning(environment, np.inf))
    agent.train()

    print(agent.learning_strategy.π)
    print()
    print(agent.learning_strategy.q_values)
    print()
    print(agent.learning_strategy.v_values)

    pi = agent.learning_strategy.π
    ideal_path = []
    # loop through the nested array pi breath first and find the largest value for each state
    for i in range(len(pi[0])):
        one = pi[0][i]
        two = pi[1][i]
        three = pi[2][i]
        four = pi[3][i]
        l = [one, two, three, four]
        ideal_path.append(np.argmax(l))
    # print ideal_path formatted to look like a 4x4 grid
    direction = {0: 'left', 1: 'down', 2: 'right', 3: 'up'}
    for i in range(len(ideal_path)):
        ideal_path[i] = direction[ideal_path[i]]
    print(np.array(ideal_path).reshape(4, 4))
