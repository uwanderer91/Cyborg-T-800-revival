import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torch.utils.data import Dataset, DataLoader
import numpy as np

class NN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def save(self, filepath='model.npz'):
        torch.save(self.state_dict(), filepath)
        print(f"Модель сохранена: {filepath}")

    def load(self, filepath='model.npz'):
        self.load_state_dict(torch.load(filepath))
        print(f"Модель загружена: {filepath}")