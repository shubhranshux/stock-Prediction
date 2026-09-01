from __future__ import annotations

from pathlib import Path

from src.data import add_indicators, fetch_reliance, prepare_sequences
from src.model import n_features, save_bundle, train_lstm

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "reliance_lstm"


def main() -> None:
    print("Downloading RELIANCE.NS from Yahoo Finance...")
    raw = fetch_reliance(period="10y", ticker="RELIANCE.NS")
    featured = add_indicators(raw)
    prepared = prepare_sequences(featured, lookback=60)
    print(f"Train sequences: {len(prepared['x_train'])} | Test sequences: {len(prepared['x_test'])}")
    print("Training LSTM...")
    x_train, y_train = prepared["x_train"], prepared["y_train"]
    val_cut = max(1, int(len(x_train) * 0.9))
    model, history = train_lstm(
        x_train[:val_cut],
        y_train[:val_cut],
        x_train[val_cut:],
        y_train[val_cut:],
        lookback=60,
        n_features=n_features(),
        epochs=50,
        batch_size=32,
        learning_rate=1e-3,
    )
    save_bundle(
        MODEL_DIR,
        model,
        prepared["scaler"],
        {
            "ticker": "RELIANCE.NS",
            "lookback": 60,
            "units": 96,
            "dropout": 0.2,
            "epochs": len(history["loss"]),
        },
    )
    print(f"Saved model to {MODEL_DIR}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")


if __name__ == "__main__":
    main()
