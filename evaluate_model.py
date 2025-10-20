import torch
from env import *
import numpy as np
import NN
import time

class Evaluator:
    def __init__(self):
        self.env = VizDoomGym(render=True)

        self.model = NN.NN(
            input_channels=1,
            num_actions=7
        )

        self.model.load()
    
    def get_action(self, obs):
        obs_encoded = np.reshape(obs, (1, 1, 64, 64))
        obs_encoded = torch.from_numpy(obs_encoded).float()
        probs = self.model(obs_encoded)
        dist = torch.distributions.Categorical(logits=probs)
        action = dist.sample().item()
        return action

    def evaluate_episode(self):
        for i in range(0, 100):
            obs = self.env.reset()
            while True:
                time.sleep(0.1)

                action = self.get_action(obs)
                obs, reward, done, _ = self.env.step(action)
                self.env.render()

                if done:
                    break
            
evaluator = Evaluator()
evaluator.evaluate_episode()