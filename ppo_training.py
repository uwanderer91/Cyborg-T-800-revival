import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import NN
import keyboard
import numpy as np
from env import *
from frame_stacker import *

torch.set_printoptions(sci_mode=False)
torch.backends.cudnn.benchmark = True

class PPO:
    def __init__(self, input_channels, num_actions, lr=0.001, gamma=0.99, clip_epsilon=0.2, entropy_coef=0.01, frame_stack=2):
        self.frame_stack = frame_stack
        
        self.actor = NN.PolicyNN(input_channels*frame_stack, num_actions)
        self.critic = NN.CriticNN(input_channels*frame_stack)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actor.to(self.device)
        self.critic.to(self.device)
        
        torch.set_float32_matmul_precision('medium')
        
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=lr)
        
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef

    def get_action(self, state):
        state_encoded = np.reshape(state, (1, self.frame_stack, 64, 64))
        state_encoded = torch.from_numpy(state_encoded).float().to(self.device)
        with torch.no_grad():
            logits = self.actor(state_encoded)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.item(), probs
        
    def compute_advantages(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        
        extended_values = torch.cat([values, next_value.unsqueeze(0)])
        extended_dones = torch.cat([dones, torch.tensor([False], device=self.device)])
        
        for t in reversed(range(len(rewards))):
            nonterminal = 1 - extended_dones[t+1].float()
            delta = rewards[t] + self.gamma * extended_values[t+1] * nonterminal - extended_values[t]
            gae = delta + self.gamma * 0.95 * gae * nonterminal
            advantages.insert(0, gae)
        
        return torch.stack(advantages)
    
    def update(self, dataloader, epochs=20):
        for epoch in range(epochs):
            epoch_ev = []
            epoch_entropy = []
            
            for batch in dataloader:
                states, actions, old_log_probs, vals, rewards, dones = [x.to(self.device, non_blocking=True) for x in batch]

                values = self.critic(states).squeeze()
                
                with torch.no_grad():
                    next_value = self.critic(states[-1:]).squeeze()
                    vals_tensor = torch.tensor(vals, dtype=torch.float32)
                    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
                    dones_tensor = torch.tensor(dones, dtype=torch.bool)

                    advantages = self.compute_advantages(rewards_tensor, vals_tensor, dones_tensor, next_value)
                    returns = advantages + vals_tensor
                    advantages = (advantages - advantages.mean()) / (advantages.std())

                with torch.no_grad():
                    returns_tensor = returns
                    values_tensor = values
                    var_returns = torch.var(returns_tensor)

                    if var_returns > 0:
                        ev = 1 - torch.var(returns_tensor - values_tensor) / var_returns
                    else:
                        ev = torch.tensor(1.0)
                    epoch_ev.append(ev.item())

                logits = self.actor(states)
                new_probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(new_probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, returns)

                total_loss = actor_loss + critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                epoch_entropy.append(entropy.item())
                
            if epoch % 1 == 0:
                mean_ev = np.mean(epoch_ev)
                mean_entropy = np.mean(epoch_entropy)
                print(f'Epoch {epoch}: '
                    f'Actor Loss: {actor_loss.item():.4f}, '
                    f'Critic Loss: {critic_loss.item():.4f}, '
                    f'Explained Var: {mean_ev:.4f}, '
                    f'Entropy: {mean_entropy:.4f}')

    def to_dataloader(self, experiences, batch_size=32):
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(np.reshape(experiences['states'], (len(experiences['states']), self.frame_stack, 64, 64))),
            torch.LongTensor(experiences['actions']),
            torch.FloatTensor(experiences['log_probs']),
            torch.FloatTensor(experiences['vals']),
            torch.FloatTensor(experiences['rewards']),
            torch.BoolTensor(experiences['dones']),
        )
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
if __name__ == "__main__":

    load_actor = False
    load_critic = False
    timesteps = 1024
    batch_size = 256
    train_iter_per_set = 20
    lr = 0.001
    gamma = 0.99
    clip_epsilon = 0.2
    entropy_coef = 0.003
    frame_stack = 1

    env = VizDoomGym(render=True)
    ppo_agent = PPO(input_channels=1, num_actions=env.action_space.n, lr=lr, gamma=gamma, 
                   clip_epsilon=clip_epsilon, entropy_coef=entropy_coef, frame_stack=frame_stack)
    
    if load_actor:
        ppo_agent.actor.load()
    if load_critic:
        ppo_agent.critic.load()
    
    frame_stacker = FrameStacker(frame_stack=frame_stack)

    while True:

        episode_experience = {
            'states': [], 'actions': [], 'log_probs': [], 'vals': [],
            'rewards': [], 'dones': []
        }
        i = 0
        state = env.reset()
        
        stacked_state = frame_stacker.reset(state)

        while i < timesteps:
            if keyboard.is_pressed('s'):
                ppo_agent.critic.save()
                ppo_agent.actor.save()

            action, log_prob, _ = ppo_agent.get_action(stacked_state)
            
            critic_input = torch.from_numpy(np.reshape(stacked_state, (1, frame_stacker.frame_stack, 64, 64))).float().to(ppo_agent.device)
            val = ppo_agent.critic(critic_input).squeeze()
            
            next_state, reward, done, _ = env.step(action)
            
            stacked_next_state = frame_stacker.append(next_state)
            
            episode_experience['states'].append(stacked_state)
            episode_experience['actions'].append(action)
            episode_experience['log_probs'].append(log_prob)
            episode_experience['vals'].append(val.detach().item())
            episode_experience['rewards'].append(reward)
            episode_experience['dones'].append(done)
            
            stacked_state = stacked_next_state
            i += 1

            if done:
                state = env.reset()
                stacked_state = frame_stacker.reset(state)

        dataloader = ppo_agent.to_dataloader(episode_experience, batch_size=batch_size)
        ppo_agent.update(dataloader, epochs=train_iter_per_set)