"""Clasificadores de ventanas (B, L, F) con salida logit (B,)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GRUClassifier(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2, **_):
        super().__init__()
        self.encoder = nn.GRU(n_features, hidden, layers, batch_first=True,
                              dropout=dropout if layers > 1 else 0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        _, hidden = self.encoder(x)
        return self.head(hidden[-1]).squeeze(-1)


class CausalConv1d(nn.Conv1d):
    """Padding exclusivamente izquierdo: la salida t nunca ve entradas > t."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size, dilation=dilation)
        self.left_padding = (kernel_size - 1) * dilation

    def forward(self, x):
        return super().forward(F.pad(x, (self.left_padding, 0)))


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels_in, hidden, dilation, dropout, kernel_size):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels_in, hidden, kernel_size, dilation), nn.GELU(),
            nn.Dropout(dropout), CausalConv1d(hidden, hidden, kernel_size, dilation),
            nn.GELU(), nn.Dropout(dropout))
        self.skip = nn.Conv1d(channels_in, hidden, 1) if channels_in != hidden else nn.Identity()

    def forward(self, x):
        return F.gelu(self.net(x) + self.skip(x))


class TCNClassifier(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2, kernel_size=3, **_):
        super().__init__()
        self.encoder = nn.Sequential(*[
            ResidualTCNBlock(n_features if i == 0 else hidden, hidden, 2 ** i, dropout, kernel_size)
            for i in range(layers)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.encoder(x.transpose(1, 2))[:, :, -1]).squeeze(-1)


class TransformerClassifier(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2, window=64, heads=4, **_):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden debe ser divisible por heads (por defecto 4).")
        self.projection = nn.Linear(n_features, hidden)
        self.position = nn.Parameter(torch.empty(1, window, hidden))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x):
        length = x.shape[1]
        if length > self.position.shape[1]:
            raise ValueError("La ventana excede el embedding posicional del modelo.")
        x = self.projection(x) + self.position[:, :length]
        mask = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        return self.head(self.encoder(x, mask=mask)[:, -1]).squeeze(-1)


def build_model(name: str, n_features: int, **hp) -> nn.Module:
    if n_features < 1 or hp.get("hidden", 64) < 1 or hp.get("layers", 2) < 1:
        raise ValueError("n_features, hidden y layers deben ser positivos.")
    if hp.get("window", 64) < 1 or not 0 <= hp.get("dropout", 0.2) < 1:
        raise ValueError("Se requiere window > 0 y 0 <= dropout < 1.")
    models = {"gru": GRUClassifier, "tcn": TCNClassifier, "transformer": TransformerClassifier}
    if name not in models:
        raise ValueError(f"Modelo desconocido: {name}")
    return models[name](n_features, **hp)
