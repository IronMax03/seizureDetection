import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMBlock(nn.Module):
    """Conv -> BatchNorm -> LeakyReLU -> FiLM modulation -> MaxPool.
    FiLM: h = gamma * h + beta, with gamma/beta per-channel, broadcast over time."""
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.LeakyReLU(0.01)
        self.pool = nn.MaxPool1d(2, stride=2)
        self.out_ch = out_ch

    def forward(self, x, gamma, beta):
        h = self.act(self.bn(self.conv(x)))
        # gamma, beta: (N, out_ch) -> (N, out_ch, 1) to broadcast over length
        h = gamma.unsqueeze(-1) * h + beta.unsqueeze(-1)
        return self.pool(h)



# First architecture: FiLM CNN, with a FiLM generator that takes extra features as input.
class FiCNN(nn.Module):
    def __init__(self, extra_features=23, n_classes=23):
        super().__init__()
        self.channels = [18, 15, 10, 10, 10]
        kernels = [6, 5, 4, 4, 4]
        in_chs = [23, 18, 15, 10, 10]

        self.blocks = nn.ModuleList([
            FiLMBlock(ic, oc, k) for ic, oc, k in zip(in_chs, self.channels, kernels)
        ])

        # FiLM generator: features -> (gamma, beta) for every channel of every block.
        total_ch = sum(self.channels)
        self.film_dim = 2 * total_ch

        # if there are no extra features, fall back to a learnable constant input
        self.no_features = (extra_features == 0)

        # FiLM generator Object: a small MLP that outputs gamma and beta for each channel in the CNN blocks.
        gen_in = extra_features if extra_features > 0 else 1
        self.film_generator = nn.Sequential(
            nn.Linear(gen_in, 32), nn.LeakyReLU(0.01),
            nn.Linear(32, self.film_dim),
        )

        # init the final layer so gamma starts at 1 and beta at 0 (identity FiLM)
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)

        # Dense layers for classification
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(40, 50), nn.LeakyReLU(0.01),   # 10 ch x 4 samples for L=256
            nn.Linear(50, 20), nn.LeakyReLU(0.01),
            nn.Linear(20, n_classes),
        )

    def forward(self, x_serie, x_features):
        N = x_serie.size(0)
        if self.no_features:
            cond = torch.ones(N, 1, device=x_serie.device, dtype=x_serie.dtype)
        else:
            cond = x_features

        film = self.film_generator(cond)          # (N, film_dim)
        idx = 0
        z = x_serie
        for blk in self.blocks:
            c = blk.out_ch
            gamma = 1.0 + film[:, idx:idx + c]
            idx += c

            beta = film[:, idx:idx + c]
            idx += c

            z = blk(z, gamma, beta)

        z = self.flatten(z)
        return self.classifier(z)


class control(FiCNN):
    def __init__(self, n_classes=2):
        super().__init__(extra_features=1, n_classes=n_classes)


class FiCRNN(nn.Module):
    pass
