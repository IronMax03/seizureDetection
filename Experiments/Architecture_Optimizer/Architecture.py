import torch
import torch.nn as nn

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

class FiCRNN_GRU_OPT(nn.Module):
    """
    This is a optimized version of FiCRNN_GRU.
    """
    def __init__(self, extra_features=0, n_classes=23, n_channels=23,
                film_generator_layers:int = 2,
                classifier_hiden: int = 16,
                channels:list[int] = [18, 15, 10, 10, 10],
                kernels:list[int]  = [6, 5, 4, 4, 4],
                dropouts_film_cnn:float = 0.1,
                dropouts_classifier:float = 0.3,
                dropout_film_generator:float = 0.2):
        super().__init__()

        # compute flattened size dynamically so the model works for any window length
        L = 256  # CHB-MIT 1-second windows at 256 Hz; change here if your data differs
        for k in kernels:
            L = (L - k + 1) // 2
        assert L > 0, f"Window length 256 is too short for this architecture (kernels {self.kernels})"
        flat_size = channels[-1] * L  # = 40 for L=256

        # instantiate FiLM generator
        blocks = []
        gen_in    = extra_features if extra_features > 0 else 1
        for i in range(film_generator_layers - 1):
            blocks.append(nn.Dropout(dropout_film_generator))
            blocks.append(nn.Linear(gen_in, 32))
            blocks.append(nn.LeakyReLU(0.01))

        self.film_dim  = 2 * sum(channels)
        blocks.append(nn.Linear(32, self.film_dim))
        self.film_generator = nn.ModuleList(blocks)
        nn.init.zeros_(self.film_generator.weight)
        nn.init.zeros_(self.film_generator.bias)

        # instantiate FiLM Layers
        channels.insert(0, n_channels)
        blocks = []
        for i, k in enumerate(kernels):
            blocks.append(FiLMBlock(channels[i], channels[i + 1] ,k, dropout_p = dropouts_film_cnn))

        self.FiLM = nn.ModuleList(blocks)

        # instantiate Classifier layers
        rnn = nn.GRU(input_size=flat_size, hidden_size=classifier_hiden, dropout = dropouts_classifier, batch_first=True)

        self.classifier = nn.Sequential(
                    rnn,
                    nn.Dropout(dropouts_classifier),
                    nn.Linear(1, n_classes),
                )

    def forward(self, x_serie, x_features):
        if x_features is not None:
            bg = self.film_generator(x_features)
            beta, gamma = bg[0:self.film_dim//2, -1], 1 + bg[self.film_dim//2:-1, -1]
        else:
            gamma, beta = [1] * (self.film_dim//2), [0] * (self.film_dim//2)

        return self.classifier(self.FiLM(x_serie, gamma, beta))
