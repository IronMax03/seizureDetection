import torch
import torch.nn as nn
import torch.nn.functional as F

class Arch1(nn.Module):
    def __init__(self, samples=4*500, random_seed=123):
        super().__init__()
        torch.manual_seed(random_seed)

        # --- Convolutional block 1 ---
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=100, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)

        self.bn1 = nn.BatchNorm1d(10)

        # --- Convolutional block 2 ---
        self.conv2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=100, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv2.bias)
        self.bn2 = nn.BatchNorm1d(1)

        # --- Fully connected layers ---
        self.hidden = nn.Linear(samples, samples)
        nn.init.kaiming_normal_(self.hidden.weight, nonlinearity='relu')
        nn.init.zeros_(self.hidden.bias)

        # --- Output layer (was referenced in forward but never defined) ---
        self.output = nn.Linear(samples, samples)
        nn.init.kaiming_normal_(self.output.weight, nonlinearity='relu')
        nn.init.zeros_(self.output.bias)

    def forward(self, x):

        # Conv1d expects (batch, channels, length); add channel axis if missing
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = x.squeeze(1)  # back to (batch, length) for the Linear layers

        # FC layers
        a_h1 = F.relu(self.hidden(x))
        return a_h1, torch.softmax(self.output(F.relu(self.hidden(a_h1))), dim=1)


        a_out = torch.softmax(self.output(a_h1), dim=1)

        return a_h1, a_out