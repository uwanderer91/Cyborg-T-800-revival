from env import *
import time
import keyboard
import data_loader

class ExpertRecorder:
    def __init__(self, env, key_to_action, delay_needed=True, subtask_label="none"):
        self.env = env
        self.expert_data = {
            'num_actions': 0,
            'observations': [],
            'actions': []
        }
        self.key_to_action = key_to_action
        self.delay_needed = delay_needed
        self.subtask_label = subtask_label

    def get_action(self):
        for key, action in self.key_to_action.items():
            if keyboard.is_pressed(key):
                return action
        return 0

    def record_episode(self, episodes):
        episode_data = []
        for i in range(0, episodes):
            obs = self.env.reset()
            while True:
                if(self.delay_needed):
                    time.sleep(0.05)

                action = self.get_action()
                episode_data.append((obs, action))
                obs, reward, done, _ = self.env.step(action)
                print(reward)
                self.env.render()

                if done:
                    break
            
        self.expert_data['num_actions'] = self.env.action_space.n
        for obs, action in episode_data:
            self.expert_data['observations'].append(obs)
            self.expert_data['actions'].append(action)

        print("records: "+str(len(self.expert_data['actions'])))
        data_loader.save_map(self.expert_data, "data_"+self.subtask_label+".npz")

if __name__ == "__main__":
    key_to_action = {
        'space': 3,
        'w': 4,
        's': 5,
        'a': 6,
        'd': 7
    }
    delay_needed = True
    num_of_games = 5
    subtask_label = "shooting"
    env = VizDoomGym(render=True)
    expert_rec = ExpertRecorder(env, key_to_action=key_to_action, delay_needed=delay_needed, subtask_label=subtask_label)
    expert_rec.record_episode(num_of_games)
    