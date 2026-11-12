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
        assert L > 0, f"Window length 256 is too short for this architecture (kernels {kernels})"
        flat_size = channels[-1] * L  # = 40 for L=256

        # instantiate FiLM generator
        blocks = []
        gen_in    = extra_features if extra_features > 0 else 1
        for i in range(film_generator_layers - 1):
            blocks.append(nn.Dropout(dropout_film_generator))
            blocks.append(nn.Linear(gen_in, 32))
            blocks.append(nn.LeakyReLU(0.01))
            gen_in = 32

        self.film_dim  = 2 * sum(channels)
        blocks.append(nn.Linear(gen_in, self.film_dim))
        self.film_generator = nn.Sequential(*blocks)
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)

        # instantiate FiLM Layers
        channels = [n_channels] + channels
        blocks = []
        for i, k in enumerate(kernels):
            blocks.append(FiLMBlock(channels[i], channels[i + 1] ,k, dropout_p = dropouts_film_cnn))

        self.FiLM = nn.ModuleList(blocks)

        # instantiate Classifier layers
        # no GRU-internal dropout: it only applies between stacked layers (num_layers=1 here)
        rnn = nn.GRU(input_size=flat_size, hidden_size=classifier_hiden, batch_first=True)

        self.classifier = nn.Sequential(
                    rnn,
                    nn.Dropout(dropouts_classifier),
                    nn.Linear(classifier_hiden, n_classes),
                )

    def forward(self, x_serie, x_features):
        sequence = x_serie.dim() == 4
        if sequence:
            batch, steps = x_serie.shape[:2]
            x_serie = x_serie.reshape(batch * steps, *x_serie.shape[2:])
            if x_features is not None:
                x_features = x_features.reshape(batch * steps, x_features.shape[-1])

        half = self.film_dim // 2
        if x_features is not None:
            bg = self.film_generator(x_features)
            beta, gamma = bg[:, :half], 1 + bg[:, half:]
        else:
            gamma, beta = x_serie.new_ones(len(x_serie), half), x_serie.new_zeros(len(x_serie), half)

        h, i = x_serie, 0
        for block in self.FiLM:
            h = block(h, gamma[:, i:i + block.out_ch], beta[:, i:i + block.out_ch])
            i += block.out_ch
        h = h.flatten(1)
        if sequence:
            # GRU runs over the window sequence: (B, T, flat) -> (B, T, hidden) -> (B, T, K)
            out, _ = self.classifier[0](h.reshape(batch, steps, -1))
            return self.classifier[1:](out)
        out, _ = self.classifier[0](h.unsqueeze(1))  # single window = length-1 sequence
        return self.classifier[1:](out[:, -1])
