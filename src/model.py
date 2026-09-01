from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import FEATURE_COLUMNS, invert_close

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StockLSTM(nn.Module):
    def __init__(self, n_features: int, units: int = 96, dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, units, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(units, units // 2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(units // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])
        return self.head(out)


def _to_loader(x, y, batch_size: int, shuffle: bool) -> DataLoader:
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=shuffle)


def train_lstm(
    x_train,
    y_train,
    x_val,
    y_val,
    lookback: int,
    n_features: int,
    units: int = 96,
    dropout: float = 0.2,
    epochs: int = 25,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
):
    del lookback  # shape comes from the arrays
    model = StockLSTM(n_features=n_features, units=units, dropout=dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-5)
    loss_fn = nn.MSELoss()

    train_loader = _to_loader(x_train, y_train, batch_size, True)
    val_loader = _to_loader(x_val, y_val, batch_size, False)

    history = {"loss": [], "val_loss": []}
    best_state = None
    best_val = float("inf")
    patience = 6
    stale = 0

    for _ in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_losses.append(float(loss_fn(model(xb), yb).item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


@torch.no_grad()
def predict_array(model: StockLSTM, x: np.ndarray) -> np.ndarray:
    model.eval()
    xt = torch.tensor(x, dtype=torch.float32, device=DEVICE)
    out = model(xt).cpu().numpy().reshape(-1)
    return out


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100)
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


@torch.no_grad()
def forecast_next_days(model, scaled_window: np.ndarray, scaler, days: int) -> np.ndarray:
    model.eval()
    window = scaled_window.copy()
    preds = []
    for _ in range(days):
        xt = torch.tensor(window[np.newaxis, ...], dtype=torch.float32, device=DEVICE)
        next_scaled_close = float(model(xt).cpu().numpy()[0, 0])
        preds.append(next_scaled_close)
        next_row = window[-1].copy()
        next_row[0] = next_scaled_close
        window = np.vstack([window[1:], next_row])
    return invert_close(scaler, np.array(preds))


def save_bundle(path: Path, model, scaler, meta: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "lstm.pt")
    joblib.dump(scaler, path / "scaler.pkl")
    joblib.dump(meta, path / "meta.pkl")


def load_bundle(path: Path):
    meta = joblib.load(path / "meta.pkl")
    model = StockLSTM(
        n_features=len(FEATURE_COLUMNS),
        units=int(meta.get("units", 96)),
        dropout=float(meta.get("dropout", 0.2)),
    )
    state = torch.load(path / "lstm.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    scaler = joblib.load(path / "scaler.pkl")
    return model, scaler, meta


def n_features() -> int:
    return len(FEATURE_COLUMNS)
