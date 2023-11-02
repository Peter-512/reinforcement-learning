from agent.agent import TabularAgent, Agent
from environment.markovdecisionprocess import MarkovDecisionProcess
from environment.openai import FrozenLakeEnvironment
from learning.tabular.qlearning import NStepQlearning, MonteCarloLearning, Qlearning
from learning.tabular.tabular_learning import TabularLearner

if __name__ == '__main__':
    # example use of the code base
    environment = FrozenLakeEnvironment(render=False, is_slippery=True, random=False, size=4)
    print(environment.map)

    # create a learning strategy
    # learning_strategy: TabularLearner = Qlearning(environment, α=0.5, γ=0.99, ε_min=0.000005)
    learning_strategy: TabularLearner = Qlearning(environment, α=0.5, γ=0.99, ε_min=0.000005, ε_max=0.5)
    # learning_strategy: TabularLearner = NStepQlearning(environment, 5)
    # learning_strategy: TabularLearner = MonteCarloLearning(environment, α=0.5, γ=0.95, ε_min=0.00005, ε_max=0.7)

    # create an Agent that uses NStepQlearning Strategy
    agent: Agent = TabularAgent(environment, learning_strategy, n_episodes=10_000)
    agent.train()

    agent.learning_strategy.show_policy()
