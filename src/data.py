from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import ta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLUMNS = [
    "Close",
    "Volume",
    "sma_20",
    "sma_50",
    "ema_12",
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_high",
    "bb_low",
    "atr_14",
    "obv",
]


def fetch_reliance(period: str = "10y", ticker: str = "RELIANCE.NS") -> pd.DataFrame:
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for {ticker}. Check the ticker or your network.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.rename(columns=str.title)
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Downloaded data is missing columns: {missing}")

    data = data[list(required)].copy()
    data = data.dropna()
    data.index.name = "Date"
    return data


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    volume = out["Volume"]

    out["sma_20"] = ta.trend.sma_indicator(close, window=20)
    out["sma_50"] = ta.trend.sma_indicator(close, window=50)
    out["ema_12"] = ta.trend.ema_indicator(close, window=12)
    out["rsi_14"] = ta.momentum.rsi(close, window=14)
    macd = ta.trend.MACD(close)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"] = bb.bollinger_lband()
    out["atr_14"] = ta.volatility.average_true_range(high, low, close, window=14)
    out["obv"] = ta.volume.on_balance_volume(close, volume)

    out = out.dropna()
    return out


def prepare_sequences(
    featured: pd.DataFrame,
    lookback: int = 60,
    test_ratio: float = 0.15,
):
    values = featured[FEATURE_COLUMNS].astype(float).values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    x, y = [], []
    for i in range(lookback, len(scaled)):
        x.append(scaled[i - lookback : i])
        y.append(scaled[i, 0])

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    split = int(len(x) * (1 - test_ratio))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    dates = featured.index[lookback:]
    train_dates = dates[:split]
    test_dates = dates[split:]

    return {
        "scaler": scaler,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "scaled": scaled,
        "featured": featured,
    }


def invert_close(scaler: MinMaxScaler, close_scaled: np.ndarray) -> np.ndarray:
    dummy = np.zeros((len(close_scaled), len(FEATURE_COLUMNS)))
    dummy[:, 0] = close_scaled.reshape(-1)
    return scaler.inverse_transform(dummy)[:, 0]


def cache_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
