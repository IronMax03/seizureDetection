import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMBlock(nn.Module):
    """Conv -> BatchNorm -> FiLM modulation -> LeakyReLU -> Dropout -> MaxPool."""
    def __init__(self, in_ch, out_ch, kernel_size, dropout_p: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=1, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch, affine=False)
        self.act  = nn.LeakyReLU(0.01)
        self.pool = nn.MaxPool1d(2, stride=2)
        self.dp   = nn.Dropout1d(p=dropout_p)
        self.out_ch = out_ch

    def forward(self, x, gamma, beta):
        h = self.bn(self.conv(x))
        h = gamma.unsqueeze(-1) * h + beta.unsqueeze(-1)  # FiLM
        h = self.dp(self.act(h))
        return self.pool(h)


class FiCNN(nn.Module):
    """FiLM-CNN conditioned on per-channel ApEn features (or unconditioned control)."""
    def __init__(self, extra_features=0, n_classes=23, n_channels=23):
        super().__init__()
        self.channels = [18, 15, 10, 10, 10]
        self.kernels  = [6, 5, 4, 4, 4]
        self.in_chs   = [n_channels, 18, 15, 10, 10]

        self.blocks = nn.ModuleList([
            FiLMBlock(ic, oc, k)
            for ic, oc, k in zip(self.in_chs, self.channels, self.kernels)
        ])

        total_ch       = sum(self.channels)
        self.film_dim  = 2 * total_ch
        self.no_features = (extra_features == 0)
        self.gen_in    = extra_features if extra_features > 0 else 1   # stored for subclasses

        self.film_generator = nn.Sequential(
            nn.Linear(self.gen_in, 32), nn.LeakyReLU(0.01),
            nn.Linear(32, self.film_dim),
        )
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)

        self.flatten    = nn.Flatten()

        # compute flattened size dynamically so the model works for any window length
        L = 256  # CHB-MIT 1-second windows at 256 Hz; change here if your data differs
        for k in self.kernels:
            L = (L - k + 1) // 2
        assert L > 0, f"Window length 256 is too short for this architecture (kernels {self.kernels})"
        self.flat_size = self.channels[-1] * L  # = 40 for L=256

        self.classifier = nn.Sequential(
            nn.Linear(self.flat_size, 50), nn.LeakyReLU(0.01),
            nn.Linear(50, 20), nn.LeakyReLU(0.01),
            nn.Linear(20, n_classes),
        )

    def forward(self, x_serie, x_features):
        N = x_serie.size(0)
        cond = (torch.ones(N, 1, device=x_serie.device, dtype=x_serie.dtype)
                if self.no_features else x_features)
        film = self.film_generator(cond)
        idx, z = 0, x_serie
        for blk in self.blocks:
            c = blk.out_ch
            gamma = 1.0 + film[:, idx:idx + c]; idx += c
            beta  =       film[:, idx:idx + c]; idx += c
            z = blk(z, gamma, beta)
        return self.classifier(self.flatten(z))


class FiCNN_reg(FiCNN):
    """FiCNN with dropout regularisation in conv blocks, FiLM generator and classifier."""
    def __init__(self, extra_features=0, n_classes=23, n_channels=23):
        super().__init__(extra_features, n_classes, n_channels)

        self.blocks = nn.ModuleList([
            FiLMBlock(ic, oc, k, dropout_p=0.1)
            for ic, oc, k in zip(self.in_chs, self.channels, self.kernels)
        ])
        self.film_generator = nn.Sequential(
            nn.Linear(self.gen_in, 32), nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(32, self.film_dim),
        )
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.flat_size, 16), nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(16, n_classes),
        )


class FiCRNN(nn.Module):
    """
    FiLM-CNN feature extractor followed by an RNN over a sequence of windows.

    Expects input as a sequence of T windows per sample:
      x_seq : (N, T, 23, W)   — EEG windows
      f_seq : (N, T, F)        — conditioning features (or zeros if F=0)

    Returns logits : (N, T, n_classes)
    The hidden state is reset to zero at the start of every sequence (stateless / Option A).
    """
    def __init__(self, extra_features=0, n_classes=23, hidden=16, n_channels=23):
        super().__init__()
        self.cnn = FiCNN_reg(extra_features, n_classes, n_channels)
        # RNN input size = flattened CNN feature vector (40 for 256-sample windows)
        self.rnn = nn.RNN(input_size=self.cnn.flat_size, hidden_size=hidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def _cnn_features(self, x, f):
        """Run the FiLM-CNN blocks and return the flat feature vector (B, 40), no classifier."""
        B = x.size(0)
        cond = (torch.ones(B, 1, device=x.device, dtype=x.dtype)
                if self.cnn.no_features else f)
        film = self.cnn.film_generator(cond)
        idx, z = 0, x
        for blk in self.cnn.blocks:
            c = blk.out_ch
            gamma = 1.0 + film[:, idx:idx + c]; idx += c
            beta  =       film[:, idx:idx + c]; idx += c
            z = blk(z, gamma, beta)
        return self.cnn.flatten(z)                           # (B, 40)

    def forward(self, x_seq, f_seq):
        # x_seq: (N, T, 23, W)   f_seq: (N, T, F) or None
        N, T = x_seq.shape[:2]
        x = x_seq.reshape(N * T, *x_seq.shape[2:])          # (N*T, 23, W)
        f = (f_seq.reshape(N * T, f_seq.shape[-1])
             if f_seq is not None else None)                  # (N*T, F) or None

        z   = self._cnn_features(x, f).reshape(N, T, -1)    # (N, T, 40)
        out, _ = self.rnn(z)                                 # (N, T, hidden); h0=None = zeros
        return self.classifier(out)                          # (N, T, n_classes)


class FiCRNN_GRU(FiCRNN):
    """
    FiLM-CNN feature extractor followed by an GRU over a sequence of windows.

    Expects input as a sequence of T windows per sample:
      x_seq : (N, T, 23, W)   — EEG windows
      f_seq : (N, T, F)        — conditioning features (or zeros if F=0)

    Returns logits : (N, T, n_classes)
    The hidden state is reset to zero at the start of every sequence (stateless / Option A).
    """
    def __init__(self, extra_features=0, n_classes=23, hidden=16, n_channels=23):
        super().__init__(extra_features, n_classes, hidden, n_channels)
        self.rnn = nn.GRU(input_size=self.cnn.flat_size, hidden_size=hidden, batch_first=True)
