import torch
import torch.nn.functional as F
from env import *
import numpy as np
import NN
import time
from frame_stacker import *

class Evaluator:
    def __init__(self, env, subtask_label, frame_stack):
        self.env = env
        self.frame_stack = frame_stack

        self.model = NN.PolicyNN(
            input_channels=1*frame_stack,
            num_actions=self.env.action_space.n
        )

        self.model.load("actor_model.npz")
    
    def get_action(self, obs):
        obs_encoded = np.reshape(obs, (1, self.frame_stack, 64, 64))
        obs_encoded = torch.from_numpy(obs_encoded).float()
        logits = self.model(obs_encoded)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        return action

    def evaluate_episode(self):
        frame_stacker = FrameStacker(frame_stack=self.frame_stack)

        for i in range(0, 100):
            obs = self.env.reset()
            frame_stacker.reset(obs)

            while True:
                time.sleep(0.05)

                action = self.get_action(frame_stacker.get_stacked_frames())
                obs, reward, done, _ = self.env.step(action)
                frame_stacker.append(obs)
                
                self.env.render()

                if done:
                    break

if __name__ == "__main__":
    subtask_label = "shooting"
    frame_stack = 1
    env = VizDoomGym(render=True)
    evaluator = Evaluator(env, subtask_label, frame_stack)
    evaluator.evaluate_episode()