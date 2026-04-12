from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class LottoLSTM(nn.Module):
    """LSTM over multi-hot rounds; predicts next-round appearance logits (45)."""

    def __init__(
        self,
        input_dim: int = 45,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


def train_lstm(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 120,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str | None = None,
    seed: int = 42,
) -> LottoLSTM:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LottoLSTM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)
    n = xt.size(0)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch_x = xt[idx]
            batch_y = yt[idx]
            opt.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict_next_probs(
    model: LottoLSTM,
    last_sequence: np.ndarray,
    device: str | None = None,
) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(last_sequence[None, ...]).to(device)
    logits = model(x)
    probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    return probs.astype(np.float64)
