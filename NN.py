import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PolicyNN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Второй блок
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Третий блок
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            
            # Дополнительный блок для лучшего извлечения признаков
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, num_actions)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def save(self, filepath='actor_model.npz'):
        torch.save(self.state_dict(), filepath)
        print(f"Actor model saved: {filepath}")

    def load(self, filepath='actor_model.npz'):
        self.load_state_dict(torch.load(filepath))
        print(f"Actor model loaded: {filepath}")

class CriticNN(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Второй блок
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Третий блок
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            
            # Дополнительный блок для лучшего извлечения признаков
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(128 * 2 * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.value_head(x)
        return x
    
    def save(self, filepath='critic_model.npz'):
        torch.save(self.state_dict(), filepath)
        print(f"Critic model saved: {filepath}")

    def load(self, filepath='critic_model.npz'):
        self.load_state_dict(torch.load(filepath))
        print(f"Critic model loaded: {filepath}")