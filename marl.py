import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOTrainer
from data_extraction import load_data
from data_preprocessing import preprocess_data

ray.init()

class DataPipelineEnv:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.state = {"extraction_done": False, "preprocessing_done": False}
        return self.state

    def step(self, action):
        if action == 0:
            load_data()
            self.state["extraction_done"] = True
        elif action == 1 and self.state["extraction_done"]:
            preprocess_data()
            self.state["preprocessing_done"] = True
        
        reward = 1 if self.state["preprocessing_done"] else 0
        done = self.state["preprocessing_done"]
        
        return self.state, reward, done, {}

config = {"env": DataPipelineEnv, "num_workers": 1, "framework": "torch"}
trainer = PPOTrainer(config=config)

for i in range(10):
    result = trainer.train()
    print(f"Iteration {i}: Reward = {result['episode_reward_mean']}")
