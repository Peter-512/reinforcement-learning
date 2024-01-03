import pickle


class SavePoint:
    def __init__(self, dir_name):
        base_name = 'savepoints'
        self.dir_name = f"{base_name}/{dir_name}"

    def save(self, agent):
        self.save_model(agent)
        self.save_replay_memory(agent)

    def save_model(self, agent):
        model_name = self.dir_name + '/model.keras'
        agent.learning_strategy.q1.save(model_name)
        agent.learning_strategy.q2.save(model_name)

    def save_replay_memory(self, agent):
        replay_memory_name = self.dir_name + '/replay_memory.pickle'
        with open(replay_memory_name, 'wb') as f:
            pickle.dump(agent.learning_strategy.replay_memory, f)
