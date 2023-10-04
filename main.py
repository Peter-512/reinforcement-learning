from agent.agent import TabularAgent, Agent
from environment.markovdecisionprocess import MarkovDecisionProcess
from environment.openai import FrozenLakeEnvironment
from learning.tabular.qlearning import NStepQlearning, MonteCarloLearning

if __name__ == '__main__':
    # example use of the code base
    environment = FrozenLakeEnvironment(render=False, is_slippery=False)
    # test_env = FrozenLakeEnvironment(render=True, is_slippery=False)
    # environment = CartPoleEnvironment(render=False)

    # create an Agent that uses Qlearning Strategy
    # agent: Agent = TabularAgent(environment, Qlearning(environment))
    # agent.train()

    # create an Agent that uses NStepQlearning Strategy
    # agent: Agent = TabularAgent(environment, Qlearning(environment))
    agent: Agent = TabularAgent(environment, NStepQlearning(environment, 5))
    # agent: Agent = TabularAgent(environment, MonteCarloLearning(environment, np.inf))
    agent.train()

    agent.learning_strategy.show_policy()
