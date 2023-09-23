from agent.agent import TabularAgent, Agent
from environment.openai import FrozenLakeEnvironment
from learning.tabular.qlearning import Qlearning, NStepQlearning

if __name__ == '__main__':
    # example use of the code base
    environment = FrozenLakeEnvironment()

    # create an Agent that uses Qlearning Strategy
    agent: Agent = TabularAgent(environment, Qlearning(environment))
    agent.train()

    # create an Agent that uses NStepQlearning Strategy
    # agent: Agent = TabularAgent(environment, NStepQlearning(environment, 5))
    # agent.train()
