
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_features=4097, random_seed=123):
        super().__init__()
        torch.manual_seed(random_seed)

        # --- Convolutional layers ---
        self.features = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=6, stride=1),   nn.LeakyReLU(0.01),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(4, 4, kernel_size=5, stride=1),   nn.LeakyReLU(0.01),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(4, 10, kernel_size=4, stride=1),  nn.LeakyReLU(0.01),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(10, 10, kernel_size=4, stride=1), nn.LeakyReLU(0.01),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(10, 15, kernel_size=4, stride=1), nn.LeakyReLU(0.01),
            nn.MaxPool1d(2, stride=2),                                        
        )

        # --- Fully-connected Layers ---
        self.classifier = nn.Sequential(
            nn.Flatten(),                      
            nn.Linear(1875, 50), nn.LeakyReLU(0.01),
            nn.Linear(50, 20),   nn.LeakyReLU(0.01),
            nn.Linear(20, 3),            
        )

    def forward(self, x):        # x: (batch, 1, 4097)
        return self.classifier(self.features(x))