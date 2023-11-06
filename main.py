from agent.agent import TabularAgent, Agent, SuperAgent
from environment.markovdecisionprocess import MarkovDecisionProcess
from environment.openai import FrozenLakeEnvironment
from learning.tabular.qlearning import NStepQlearning, MonteCarloLearning, Qlearning
from learning.tabular.tabular_learning import TabularLearner

if __name__ == '__main__':
    # example use of the code base
    environment = FrozenLakeEnvironment(render=False, is_slippery=True, random=False, size=4)
    print(environment.map)

    # create a learning strategy
    # learning_strategy: TabularLearner = Qlearning(environment)
    # learning_strategy: TabularLearner = Qlearning(environment, α=0.5, λ=0.001, γ=0.99, ε_min=0.000005, ε_max=0.5, t_max=10000)
    # learning_strategy: TabularLearner = NStepQlearning(environment, 5, α=0.5, γ=0.99, ε_min=0.000005, ε_max=0.5)
    # learning_strategy: TabularLearner = MonteCarloLearning(environment, α=0.5, λ=0.001, γ=0.90, t_max=10000)

    # create an Agent that uses a Strategy
    # agent: Agent = TabularAgent(environment, learning_strategy, n_episodes=10_000)
    # agent.train()

    # agent.learning_strategy.show_policy()

    mdp = MarkovDecisionProcess(environment)
    agent = SuperAgent(mdp)
    agent.train()
