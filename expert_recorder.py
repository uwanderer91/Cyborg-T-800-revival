from env import *
import time
import keyboard
import data_loader

class ExpertRecorder:
    def __init__(self):
        self.env = VizDoomGym(render=True)
        self.expert_data = {
            'observations': [],
            'actions': []
        }
        self.key_to_action = {
            'left': 0,
            'right': 1, 
            'space': 2,
            'up': 3,
            'down': 4,
            'a': 5,
            'd': 6,
        }

    def get_action(self):
        for key, action in self.key_to_action.items():
            if keyboard.is_pressed(key):
                return action
        return 0

    def record_episode(self):
        episode_data = []
        for i in range(0, 8):
            obs = self.env.reset()
            while True:
                time.sleep(0.1)

                action = self.get_action()
                episode_data.append((obs, action))
                obs, reward, done, _ = self.env.step(action)
                self.env.render()

                if done:
                    break
            
        for obs, action in episode_data:
            self.expert_data['observations'].append(obs)
            self.expert_data['actions'].append(action)

        data_loader.save_spectation(self.expert_data)

if __name__ == "__main__":
    expert_rec = ExpertRecorder()
    expert_rec.record_episode()
    