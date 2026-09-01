from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.data import FEATURE_COLUMNS, add_indicators, fetch_reliance, invert_close, prepare_sequences
from src.model import (
    evaluate_predictions,
    forecast_next_days,
    load_bundle,
    n_features,
    predict_array,
    save_bundle,
    train_lstm,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "reliance_lstm"
DATA_DIR = ROOT / "data"

st.set_page_config(
    page_title="Reliance LSTM Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #0b1220; color: #e8eefc; }
      [data-testid="stSidebar"] { background: #10192b; }
      h1, h2, h3 { color: #f4f7ff !important; }
      .metric-card {
        background: linear-gradient(180deg, #16233a 0%, #121b2e 100%);
        border: 1px solid #2a3d63;
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
      }
      .hero {
        background: radial-gradient(1200px 400px at 10% -20%, #1e3a8a55, transparent),
                    linear-gradient(90deg, #12203a, #0f172a);
        border: 1px solid #27406a;
        border-radius: 20px;
        padding: 22px 26px;
        margin-bottom: 12px;
      }
      .hint { color: #9db0d0; font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def candle_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="#60a5fa", opacity=0.7),
        row=2,
        col=1,
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        height=560,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def line_compare(dates, actual, predicted, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual, name="Actual close", line=dict(color="#93c5fd", width=2)))
    fig.add_trace(go.Scatter(x=dates, y=predicted, name="LSTM prediction", line=dict(color="#f59e0b", width=2)))
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        height=460,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=1.08),
        yaxis_title="INR",
    )
    return fig


@st.cache_data(show_spinner=False)
def load_market_data(ticker: str, period: str) -> pd.DataFrame:
    df = fetch_reliance(period=period, ticker=ticker)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "reliance.csv")
    return df


def model_exists() -> bool:
            return (MODEL_DIR / "lstm.pt").exists() and (MODEL_DIR / "scaler.pkl").exists()


with st.sidebar:
    st.header("Training controls")
    ticker = st.text_input("Ticker", value="RELIANCE.NS")
    period = st.selectbox("History window", ["5y", "10y", "max"], index=1)
    lookback = st.slider("Lookback days", 30, 120, 60, 5)
    units = st.select_slider("LSTM units", options=[64, 96, 128], value=96)
    dropout = st.slider("Dropout", 0.1, 0.5, 0.2, 0.05)
    epochs = st.slider("Epochs", 5, 60, 25, 5)
    batch_size = st.select_slider("Batch size", options=[16, 32, 64], value=32)
    learning_rate = st.select_slider("Learning rate", options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3, format_func=lambda x: f"{x:g}")
    forecast_days = st.slider("Forecast horizon (days)", 1, 30, 7)
    train_clicked = st.button("Train LSTM on Reliance", type="primary", use_container_width=True)
    st.caption("Uses Yahoo Finance data for Reliance Industries (NSE: RELIANCE.NS).")

st.markdown(
    """
    <div class="hero">
      <h1>Reliance Stock LSTM Predictor</h1>
      <p class="hint">
        End-to-end market dashboard: live historical data, technical features, LSTM training,
        hold-out evaluation, and a multi-day close forecast for Reliance Industries.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    raw = load_market_data(ticker, period)
except Exception as exc:
    st.error(str(exc))
    st.stop()

featured = add_indicators(raw)
latest = featured.iloc[-1]
prev = featured.iloc[-2]
change = float(latest["Close"] - prev["Close"])
change_pct = change / float(prev["Close"]) * 100

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Last close", f"₹{latest['Close']:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
c2.metric("RSI (14)", f"{latest['rsi_14']:.1f}")
c3.metric("SMA 20", f"₹{latest['sma_20']:,.2f}")
c4.metric("SMA 50", f"₹{latest['sma_50']:,.2f}")
c5.metric("Volume", f"{latest['Volume']:,.0f}")

overview, train_tab, predict_tab, forecast_tab = st.tabs(
    ["Market overview", "Train LSTM", "Test predictions", "Forward forecast"]
)

with overview:
    st.plotly_chart(candle_chart(raw.tail(400), f"{ticker} candlesticks"), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.subheader("Recent sessions")
        show = featured[["Open", "High", "Low", "Close", "Volume", "rsi_14", "sma_20"]].tail(12)
        st.dataframe(show.style.format("{:.2f}"), use_container_width=True, height=380)
    with right:
        st.subheader("Feature set used by the LSTM")
        st.write(
            "The model is trained on scaled OHLCV-derived features rather than close price alone:"
        )
        st.code("\n".join(FEATURE_COLUMNS))
        st.caption(f"{len(featured):,} indicator-ready rows after dropping warmup NaNs.")

prepared = prepare_sequences(featured, lookback=lookback)
x_train, y_train = prepared["x_train"], prepared["y_train"]
x_test, y_test = prepared["x_test"], prepared["y_test"]

if train_clicked:
    with st.spinner("Training LSTM on Reliance history. This can take a few minutes..."):
        val_cut = max(1, int(len(x_train) * 0.9))
        model, history = train_lstm(
            x_train[:val_cut],
            y_train[:val_cut],
            x_train[val_cut:],
            y_train[val_cut:],
            lookback=lookback,
            n_features=n_features(),
            units=units,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        save_bundle(
            MODEL_DIR,
            model,
            prepared["scaler"],
            {
                "ticker": ticker,
                "lookback": lookback,
                "units": units,
                "dropout": dropout,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "feature_columns": FEATURE_COLUMNS,
            },
        )
        st.session_state["history"] = history
        st.session_state["just_trained"] = True
    st.success(f"Model saved to `{MODEL_DIR}`.")

with train_tab:
    st.write("Train a 2-layer LSTM on chronological Reliance data. The last 15% of sequences are held out for testing.")
    st.write(
        {
            "Train sequences": int(len(x_train)),
            "Test sequences": int(len(x_test)),
            "Lookback": lookback,
            "Features": n_features(),
        }
    )
    if "history" in st.session_state:
        hist = st.session_state["history"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=hist["loss"], name="Train loss"))
        fig.add_trace(go.Scatter(y=hist["val_loss"], name="Validation loss"))
        fig.update_layout(
            title="Training curve",
            template="plotly_dark",
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0b1220",
            height=380,
            yaxis_title="MSE",
            xaxis_title="Epoch",
        )
        st.plotly_chart(fig, use_container_width=True)
    elif model_exists():
        st.info("A saved Reliance LSTM is available. Retrain from the sidebar if you want a fresh fit.")
    else:
        st.warning("No trained model yet. Set hyperparameters in the sidebar and click Train.")

loaded = None
if model_exists():
    try:
        loaded = load_bundle(MODEL_DIR)
    except Exception as exc:
        st.error(f"Could not load saved model: {exc}")

with predict_tab:
    if loaded is None:
        st.warning("Train the model first to see hold-out predictions.")
    else:
        model, scaler, meta = loaded
        pred_scaled = predict_array(model, x_test)
        actual = invert_close(prepared["scaler"], y_test)
        predicted = invert_close(prepared["scaler"], pred_scaled)
        metrics = evaluate_predictions(actual, predicted)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RMSE", f"₹{metrics['RMSE']:.2f}")
        m2.metric("MAE", f"₹{metrics['MAE']:.2f}")
        m3.metric("MAPE", f"{metrics['MAPE']:.2f}%")
        m4.metric("R²", f"{metrics['R2']:.3f}")
        st.plotly_chart(
            line_compare(prepared["test_dates"], actual, predicted, "Hold-out test: actual vs LSTM"),
            use_container_width=True,
        )
        table = pd.DataFrame(
            {"Actual": actual, "Predicted": predicted, "Error": predicted - actual},
            index=prepared["test_dates"],
        )
        st.dataframe(table.tail(20).style.format("{:.2f}"), use_container_width=True)

with forecast_tab:
    if loaded is None:
        st.warning("Train the model first to generate a forward forecast.")
    else:
        model, scaler, meta = loaded
        last_window = prepared["scaled"][-lookback:]
        future_prices = forecast_next_days(model, last_window, prepared["scaler"], forecast_days)
        last_date = featured.index[-1]
        future_dates = pd.bdate_range(last_date + timedelta(days=1), periods=forecast_days)
        future_df = pd.DataFrame({"Forecast close (INR)": future_prices}, index=future_dates)
        future_df.index.name = "Date"

        hist_tail = featured["Close"].tail(80)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail.values, name="History", line=dict(color="#93c5fd", width=2)))
        fig.add_trace(
            go.Scatter(
                x=future_df.index,
                y=future_df["Forecast close (INR)"],
                name="LSTM forecast",
                line=dict(color="#34d399", width=3, dash="dash"),
            )
        )
        fig.update_layout(
            title=f"{forecast_days}-day Reliance close forecast",
            template="plotly_dark",
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0b1220",
            height=460,
            yaxis_title="INR",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(future_df.style.format("{:.2f}"), use_container_width=True)
        st.caption(
            "Forecasts are recursive LSTM rollouts, not trading advice. Markets are noisy; treat this as a research demo."
        )
