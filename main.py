from agent.agent import TabularAgent, Agent, ApproximateAgent
from environment.markovdecisionprocess import MarkovDecisionProcess
from environment.openai import FrozenLakeEnvironment, CliffWalkingEnvironment, CartPoleEnvironment
from learning.approximate.deep_qlearning import DeepQLearning
from learning.learningstrategy import LearningStrategy
from learning.tabular.qlearning import NStepQlearning, MonteCarloLearning, Qlearning
from learning.tabular.tabular_learning import TabularLearner

if __name__ == '__main__':
    # example use of the code base
    # environment = FrozenLakeEnvironment(render=False, is_slippery=True, random=False, size=4)
    # print(environment.map)

    # create a learning strategy
    # learning_strategy: TabularLearner = Qlearning(environment, α=0.5, λ=0.006, γ=0.995, ε_min=0.000005, ε_max=0.5,
    #                                               t_max=10000)
    # learning_strategy: TabularLearner = NStepQlearning(environment, 10, λ=0.004, α=0.5, γ=0.99, ε_min=0.000005,
    #                                                    ε_max=0.5,
    #                                                    t_max=10000)
    # learning_strategy: TabularLearner = MonteCarloLearning(environment, α=0.7, λ=0.0009, γ=0.99, ε_min=0.001,
    #                                                        ε_max=0.1, t_max=1000)

    # create an Agent that uses a Strategy
    # agent: Agent = TabularAgent(environment, learning_strategy, n_episodes=10_000)
    # agent.train()
    #
    # agent.learning_strategy.show_policy()

    print("Have you moved any screenshots you want to keep out of the screenshots folder?")
    input("Press Enter to continue...")

    use_savepoint = False
    savepoint_load_from_base_path = 'savepoints'
    savepoint_save_to_base_path = 'savepoints'

    environment = CartPoleEnvironment(render=True) if use_savepoint else CartPoleEnvironment(render=False)

    learning_strategy: LearningStrategy = DeepQLearning(environment, batch_size=64, ddqn=False, λ=0.006, γ=0.99,
                                                        t_max=10000, C=10, ϵ_min=0.000005,
                                                        ϵ_max=0.00005 if use_savepoint else 0.5, verbose=False,
                                                        use_savepoint=use_savepoint,
                                                        savepoint_base_path=savepoint_load_from_base_path)

    agent = ApproximateAgent(environment, learning_strategy, n_episodes=10_000,
                             savepoint_base_path=savepoint_save_to_base_path)

    agent.train()
