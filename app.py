from __future__ import annotations

from datetime import datetime, timedelta
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
    page_title="AlphaPulse | LSTM Stock Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Ultra-High-Contrast Dark Fintech CSS Theme
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

      * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      }

      code, pre, .mono {
        font-family: 'JetBrains Mono', monospace !important;
      }

      /* Global App Background & Base Contrast */
      .stApp {
        background: radial-gradient(circle at 15% 10%, #0f1c3a 0%, #0a1124 55%, #060a17 100%) !important;
        color: #ffffff !important;
      }

      /* Base Text Color & High-Contrast Readability */
      p, span, div, li {
        color: #f1f5f9;
      }
      h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
      }

      /* Header Bar Fix */
      header[data-testid="stHeader"] {
        background: rgba(10, 17, 36, 0.92) !important;
        backdrop-filter: blur(16px) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
      }
      header[data-testid="stHeader"] * {
        color: #cbd5e1 !important;
      }

      /* All Streamlit Form Widget Labels - Crystal Clear Visibility */
      label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.01em;
      }

      /* Sidebar Styling */
      [data-testid="stSidebar"] {
        background: #0d162d !important;
        border-right: 1px solid rgba(255, 255, 255, 0.12) !important;
      }
      [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        letter-spacing: -0.02em;
      }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #e2e8f0 !important;
      }
      [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] .stCaption * {
        color: #94a3b8 !important;
      }

      /* Input Fields in Sidebar */
      [data-testid="stSidebar"] input, 
      [data-testid="stSidebar"] select,
      [data-testid="stSidebar"] [data-baseweb="input"],
      [data-testid="stSidebar"] [data-baseweb="select"] {
        background-color: #142038 !important;
        color: #ffffff !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 10px !important;
      }
      [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #142038 !important;
        color: #ffffff !important;
        border: none !important;
      }

      /* Slider Theme */
      .stSlider div[data-testid="stThumbValue"] {
        color: #38bdf8 !important;
        font-weight: 800;
        font-size: 0.9rem;
      }

      /* Primary Action Button */
      button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #0ea5e9 50%, #38bdf8 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 0.65rem 1.25rem !important;
        box-shadow: 0 4px 20px rgba(14, 165, 233, 0.4) !important;
        transition: all 0.25s ease !important;
        letter-spacing: 0.02em;
      }
      button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 28px rgba(56, 189, 248, 0.6) !important;
      }

      /* Tabs Styling */
      [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(19, 31, 56, 0.8);
        padding: 6px 8px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        margin-bottom: 20px;
      }
      [data-testid="stTabs"] [data-baseweb="tab"] {
        height: auto;
        padding: 10px 20px !important;
        border-radius: 10px !important;
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.2s ease;
      }
      [data-testid="stTabs"] [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(2, 132, 199, 0.35) 0%, rgba(14, 165, 233, 0.25) 100%) !important;
        color: #38bdf8 !important;
        font-weight: 700 !important;
        border: 1px solid rgba(56, 189, 248, 0.45) !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3) !important;
      }

      /* Hero Container */
      .hero-box {
        background: linear-gradient(135deg, rgba(19, 32, 60, 0.9) 0%, rgba(13, 22, 44, 0.85) 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 20px;
        padding: 24px 30px;
        margin-bottom: 22px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
      }
      .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 40%, #e2e8f0 70%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
        line-height: 1.2;
      }
      .hero-subtitle {
        color: #cbd5e1 !important;
        font-size: 0.98rem;
        margin: 0;
        max-width: 860px;
        line-height: 1.5;
      }
      .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
      }
      .badge-live {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.45);
      }
      .badge-ai {
        background: rgba(56, 189, 248, 0.18);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.4);
      }
      .pulse-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10b981;
        animation: pulse 2s infinite;
      }
      @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
        100% { opacity: 1; transform: scale(1); }
      }

      /* KPI Metric Cards */
      .kpi-card {
        background: linear-gradient(180deg, rgba(20, 34, 64, 0.85) 0%, rgba(14, 24, 48, 0.92) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 16px 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, border-color 0.2s ease;
      }
      .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.5);
      }
      .kpi-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #7dd3fc !important;
        margin-bottom: 6px;
      }
      .kpi-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #ffffff !important;
        letter-spacing: -0.02em;
        line-height: 1.1;
      }
      .kpi-delta-up {
        color: #34d399 !important;
        font-size: 0.88rem;
        font-weight: 700;
        margin-top: 4px;
      }
      .kpi-delta-down {
        color: #f87171 !important;
        font-size: 0.88rem;
        font-weight: 700;
        margin-top: 4px;
      }
      .kpi-sub {
        color: #cbd5e1 !important;
        font-size: 0.82rem;
        margin-top: 4px;
        font-weight: 500;
      }

      /* Feature Pill Grid */
      .feature-pill {
        display: inline-block;
        padding: 6px 14px;
        background: rgba(22, 37, 70, 0.9);
        border: 1px solid rgba(56, 189, 248, 0.4);
        border-radius: 8px;
        font-size: 0.84rem;
        font-weight: 600;
        color: #7dd3fc !important;
        margin: 4px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def candle_chart(df: pd.DataFrame, title: str, show_overlays: bool = True) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#10b981",
            increasing_fillcolor="#10b981",
            decreasing_line_color="#f43f5e",
            decreasing_fillcolor="#f43f5e",
            whiskerwidth=0.6,
        ),
        row=1,
        col=1,
    )

    if show_overlays and "sma_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["sma_20"],
                name="SMA 20",
                line=dict(color="#38bdf8", width=1.5),
                opacity=0.9,
            ),
            row=1,
            col=1,
        )
    if show_overlays and "sma_50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["sma_50"],
                name="SMA 50",
                line=dict(color="#f59e0b", width=1.5),
                opacity=0.85,
            ),
            row=1,
            col=1,
        )
    if show_overlays and "bb_high" in df.columns and "bb_low" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["bb_high"],
                name="Bollinger High",
                line=dict(color="rgba(168, 85, 247, 0.5)", width=1, dash="dot"),
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["bb_low"],
                name="Bollinger Low",
                line=dict(color="rgba(168, 85, 247, 0.5)", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(168, 85, 247, 0.06)",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    # Volume with conditional color
    colors = [
        "#10b981" if c >= o else "#f43f5e"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.7,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0c1426",
        height=580,
        font=dict(color="#e2e8f0"),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(19, 31, 56, 0.8)",
            bordercolor="rgba(255, 255, 255, 0.15)",
            borderwidth=1,
            font=dict(size=11, color="#cbd5e1"),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="rgba(255, 255, 255, 0.08)", zerolinecolor="rgba(255, 255, 255, 0.12)")
    fig.update_yaxes(gridcolor="rgba(255, 255, 255, 0.08)", zerolinecolor="rgba(255, 255, 255, 0.12)")
    return fig


def line_compare(dates, actual, predicted, title: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=actual,
            name="Actual Close",
            line=dict(color="#38bdf8", width=2),
            mode="lines",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=predicted,
            name="LSTM Prediction",
            line=dict(color="#f59e0b", width=2, dash="solid"),
            mode="lines",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([dates, dates[::-1]]),
            y=np.concatenate([predicted, actual[::-1]]),
            fill="toself",
            fillcolor="rgba(245, 158, 11, 0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
            name="Error Band",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0c1426",
        height=480,
        font=dict(color="#e2e8f0"),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(19, 31, 56, 0.8)",
            bordercolor="rgba(255, 255, 255, 0.15)",
            borderwidth=1,
            font=dict(size=11, color="#cbd5e1"),
        ),
        yaxis=dict(title="Price (INR)", gridcolor="rgba(255, 255, 255, 0.08)"),
        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.08)"),
        hovermode="x unified",
    )
    return fig


@st.cache_data(show_spinner=False, ttl=3600)
def load_market_data(ticker: str, period: str) -> pd.DataFrame:
    df = fetch_reliance(period=period, ticker=ticker)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / f"{ticker.replace('.', '_')}.csv")
    return df


def model_exists() -> bool:
    return (MODEL_DIR / "lstm.pt").exists() and (MODEL_DIR / "scaler.pkl").exists()


# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
          <div style="width: 38px; height: 38px; border-radius: 10px; background: linear-gradient(135deg, #0284c7, #38bdf8); display: flex; align-items: center; justify-content: center; font-size: 1.25rem; box-shadow: 0 4px 16px rgba(56, 189, 248, 0.45);">
            📈
          </div>
          <div>
            <div style="font-weight: 800; font-size: 1.15rem; color: #ffffff; letter-spacing: -0.02em;">AlphaPulse</div>
            <div style="font-size: 0.72rem; color: #7dd3fc; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">LSTM Deep Learning</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ Asset & Window")
    popular_tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "TATAMOTORS.NS", "Custom..."]
    selected_choice = st.selectbox("Select Asset Ticker", popular_tickers, index=0)
    if selected_choice == "Custom...":
        ticker = st.text_input("Enter Symbol (e.g. AAPL, NVDA)", value="RELIANCE.NS")
    else:
        ticker = selected_choice

    period = st.selectbox("Historical Window", ["3y", "5y", "10y", "max"], index=2)

    st.markdown("---")
    st.markdown("### 🧠 Model Parameters")
    lookback = st.slider("Lookback Window (Days)", 30, 120, 60, 5, help="Past trading sessions fed into the LSTM sequence.")
    units = st.select_slider("LSTM Hidden Units", options=[64, 96, 128, 160], value=96)
    dropout = st.slider("Dropout Regularization", 0.1, 0.5, 0.2, 0.05)
    epochs = st.slider("Training Epochs", 5, 60, 25, 5)
    batch_size = st.select_slider("Batch Size", options=[16, 32, 64], value=32)
    learning_rate = st.select_slider(
        "Learning Rate",
        options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        value=1e-3,
        format_func=lambda x: f"{x:g}",
    )

    st.markdown("---")
    st.markdown("### 🔮 Inference Horizon")
    forecast_days = st.slider("Forward Horizon (Days)", 1, 30, 7)

    train_clicked = st.button("⚡ Retrain Neural Network", type="primary", width="stretch")

    if model_exists():
        st.markdown(
            """
            <div style="margin-top: 14px; padding: 10px 14px; border-radius: 10px; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); display: flex; align-items: center; gap: 8px;">
              <span class="pulse-dot"></span>
              <span style="font-size: 0.82rem; color: #34d399; font-weight: 700;">Active Model Loaded</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="margin-top: 14px; padding: 10px 14px; border-radius: 10px; background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.4); display: flex; align-items: center; gap: 8px;">
              <span style="color: #fbbf24; font-size: 0.82rem; font-weight: 700;">⚠ Model Not Trained</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -------------------------------------------------------------
# HERO BANNER
# -------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-box">
      <div style="display: flex; gap: 8px; margin-bottom: 12px;">
        <span class="badge badge-live"><span class="pulse-dot"></span> NSE Market Live</span>
        <span class="badge badge-ai">PyTorch 2-Layer LSTM</span>
        <span class="badge badge-ai">12 Indicator Inputs</span>
      </div>
      <h1 class="hero-title">{ticker} Stock Intelligence Engine</h1>
      <p class="hero-subtitle">
        Deep sequential neural forecasting using dual LSTM layers, non-linear dropout regularization, and multi-factor technical analysis (Momentum, Trend, Volatility, and Volume).
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# DATA LOAD & FEATURE EXTRACTION
# -------------------------------------------------------------
try:
    with st.spinner(f"Fetching real-time market data for {ticker}..."):
        raw = load_market_data(ticker, period)
except Exception as exc:
    st.error(f"Failed to fetch market data for {ticker}: {exc}")
    st.stop()

featured = add_indicators(raw)
latest = featured.iloc[-1]
prev = featured.iloc[-2]
change = float(latest["Close"] - prev["Close"])
change_pct = change / float(prev["Close"]) * 100
is_positive = change >= 0

# RSI categorization
rsi_val = float(latest["rsi_14"])
if rsi_val > 70:
    rsi_badge = "<span style='color: #f87171; font-weight: 700;'>Overbought (>70)</span>"
elif rsi_val < 30:
    rsi_badge = "<span style='color: #34d399; font-weight: 700;'>Oversold (<30)</span>"
else:
    rsi_badge = "<span style='color: #cbd5e1; font-weight: 700;'>Neutral Momentum</span>"

# SMA Trend check
sma_trend = "Bullish Alignment" if latest["sma_20"] > latest["sma_50"] else "Bearish Alignment"
sma_color = "#34d399" if latest["sma_20"] > latest["sma_50"] else "#f87171"

# -------------------------------------------------------------
# TOP METRICS ROW
# -------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    delta_class = "kpi-delta-up" if is_positive else "kpi-delta-down"
    delta_symbol = "▲" if is_positive else "▼"
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">Last Market Close</div>
          <div class="kpi-value">₹{latest['Close']:,.2f}</div>
          <div class="{delta_class}">{delta_symbol} {change:+.2f} ({change_pct:+.2f}%)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi2:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">RSI (14-Day)</div>
          <div class="kpi-value">{rsi_val:.1f}</div>
          <div class="kpi-sub">{rsi_badge}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi3:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">SMA 20 vs SMA 50</div>
          <div class="kpi-value">₹{latest['sma_20']:,.2f}</div>
          <div class="kpi-sub" style="color: {sma_color}; font-weight: 700;">{sma_trend}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi4:
    bb_dist = ((latest["Close"] - latest["bb_low"]) / (latest["bb_high"] - latest["bb_low"])) * 100
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">Bollinger Range %B</div>
          <div class="kpi-value">{bb_dist:.1f}%</div>
          <div class="kpi-sub">₹{latest['bb_low']:,.0f} — ₹{latest['bb_high']:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi5:
    vol_millions = latest["Volume"] / 1e6
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">24h Session Volume</div>
          <div class="kpi-value">{vol_millions:.2f}M</div>
          <div class="kpi-sub">{latest['Volume']:,.0f} shares</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# SEQUENCE PREPARATION & TRAINING HANDLER
# -------------------------------------------------------------
prepared = prepare_sequences(featured, lookback=lookback)
x_train, y_train = prepared["x_train"], prepared["y_train"]
x_test, y_test = prepared["x_test"], prepared["y_test"]

if train_clicked:
    with st.status(f"Training 2-Layer PyTorch LSTM on {len(x_train):,} sequences...", expanded=True) as status:
        st.write("Initializing model architecture & PyTorch optimizer...")
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
        st.write("Saving model weights and scaling parameters to disk...")
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
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        st.session_state["history"] = history
        st.session_state["just_trained"] = True
        status.update(label="Training complete! Neural network weights saved successfully.", state="complete")
    st.success("Model trained and synchronized with active session.")

# -------------------------------------------------------------
# TABS INTERFACE
# -------------------------------------------------------------
tab_market, tab_train, tab_test, tab_forecast = st.tabs(
    [
        "📊 Market Terminal",
        "🧠 Neural Network & Training",
        "🎯 Hold-Out Validation",
        "🔮 Forward Forecast",
    ]
)

# TAB 1: MARKET TERMINAL
with tab_market:
    chart_col, control_col = st.columns([4, 1])
    with chart_col:
        st.markdown("<h3 style='margin-bottom: 2px;'>Candlestick Price Action & Volume</h3>", unsafe_allow_html=True)
        st.caption("Interactive Candlestick chart with technical overlays (SMA 20, SMA 50, Bollinger Bands)")
    with control_col:
        show_indicators = st.checkbox("Technical Overlays", value=True)
        candles_count = st.selectbox("Show Last Sessions", [100, 250, 400, "All"], index=2)

    df_slice = raw if candles_count == "All" else raw.tail(int(candles_count))
    st.plotly_chart(
        candle_chart(df_slice, f"{ticker} Market Sessions", show_overlays=show_indicators),
        width="stretch",
    )

    col_recent, col_features = st.columns([1.2, 1])
    with col_recent:
        st.markdown("### 📋 Recent Trading Sessions")
        recent_df = featured[["Open", "High", "Low", "Close", "Volume", "rsi_14", "sma_20"]].tail(10)
        st.dataframe(
            recent_df.style.format(
                {
                    "Open": "₹{:.2f}",
                    "High": "₹{:.2f}",
                    "Low": "₹{:.2f}",
                    "Close": "₹{:.2f}",
                    "Volume": "{:,.0f}",
                    "rsi_14": "{:.1f}",
                    "sma_20": "₹{:.2f}",
                }
            ),
            width="stretch",
            height=340,
        )

    with col_features:
        st.markdown("### 🔬 Multi-Factor Feature Matrix")
        st.markdown(
            """
            <div style="color: #e2e8f0; font-size: 0.9rem; line-height: 1.5; margin-bottom: 12px;">
              The LSTM network ingests <b>12 multidimensional feature channels</b> simultaneously to capture non-linear market dynamics beyond raw price:
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        feature_badges_html = "".join(
            [f"<span class='feature-pill'>{feat}</span>" for feat in FEATURE_COLUMNS]
        )
        st.markdown(f"<div>{feature_badges_html}</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="margin-top: 18px; padding: 14px 18px; border-radius: 12px; background: rgba(19, 31, 56, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); font-size: 0.88rem; color: #e2e8f0;">
              ⚡ <b>Dataset Scale:</b> <span style="color: #38bdf8; font-weight: 700;">{len(featured):,}</span> fully prepared indicator rows processed with MinMaxScaler standardization.
            </div>
            """,
            unsafe_allow_html=True,
        )


# TAB 2: NEURAL NETWORK & TRAINING
with tab_train:
    st.markdown("### 🧬 PyTorch Deep Learning Architecture")
    st.markdown(
        """
        <div style="color: #cbd5e1; font-size: 0.95rem; margin-bottom: 20px;">
          Chronological temporal sequence modeling with recurrent memory cells, intermediate dropout gates, and adaptive learning rate scheduling.
        </div>
        """,
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Training Sequences</div>
              <div class="kpi-value">{len(x_train):,}</div>
              <div class="kpi-sub">85% Chronological Split</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Test Sequences</div>
              <div class="kpi-value">{len(x_test):,}</div>
              <div class="kpi-sub">15% Hold-Out Evaluation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t3:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Sequence Lookback</div>
              <div class="kpi-value">{lookback} <span style="font-size: 1rem; color: #7dd3fc;">Days</span></div>
              <div class="kpi-sub">Window Memory Depth</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t4:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Input Features</div>
              <div class="kpi-value">{n_features()} <span style="font-size: 1rem; color: #7dd3fc;">Channels</span></div>
              <div class="kpi-sub">Multivariate Vector</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Training Curve Plot
    if "history" in st.session_state:
        hist = st.session_state["history"]
        fig_hist = go.Figure()
        epochs_arr = list(range(1, len(hist["loss"]) + 1))
        fig_hist.add_trace(
            go.Scatter(
                x=epochs_arr,
                y=hist["loss"],
                name="Train Loss (MSE)",
                line=dict(color="#38bdf8", width=2.5),
            )
        )
        fig_hist.add_trace(
            go.Scatter(
                x=epochs_arr,
                y=hist["val_loss"],
                name="Validation Loss (MSE)",
                line=dict(color="#f59e0b", width=2.5, dash="dash"),
            )
        )
        fig_hist.update_layout(
            title="Loss Convergence Curve Across Training Epochs",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0c1426",
            height=400,
            font=dict(color="#e2e8f0"),
            xaxis=dict(title="Epoch", gridcolor="rgba(255, 255, 255, 0.08)"),
            yaxis=dict(title="Mean Squared Error (Scaled)", gridcolor="rgba(255, 255, 255, 0.08)"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(19, 31, 56, 0.8)",
            ),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_hist, width="stretch")
    elif model_exists():
        st.markdown(
            """
            <div style="padding: 16px 20px; border-radius: 12px; background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.35); color: #ffffff; font-size: 0.95rem;">
              ℹ <b>Trained Model Ready:</b> A pre-trained model bundle is active in memory. Click <b>'⚡ Retrain Neural Network'</b> in the sidebar whenever you wish to execute a fresh parameter optimization.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("No model found. Please click '⚡ Retrain Neural Network' in the sidebar to train.")


# TAB 3: HOLD-OUT TEST PREDICTIONS
loaded = None
if model_exists():
    try:
        loaded = load_bundle(MODEL_DIR)
    except Exception as exc:
        st.error(f"Failed loading model weights: {exc}")

with tab_test:
    if loaded is None:
        st.warning("Model bundle not found. Please train the model from the sidebar first.")
    else:
        model, scaler, meta = loaded
        pred_scaled = predict_array(model, x_test)
        actual = invert_close(prepared["scaler"], y_test)
        predicted = invert_close(prepared["scaler"], pred_scaled)
        metrics = evaluate_predictions(actual, predicted)

        # Directional Accuracy calculation
        actual_diff = np.diff(actual)
        pred_diff = np.diff(predicted)
        dir_acc = (np.sign(actual_diff) == np.sign(pred_diff)).mean() * 100

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">Root Mean Sq Error</div>
                  <div class="kpi-value">₹{metrics['RMSE']:.2f}</div>
                  <div class="kpi-sub">Rupee Standard Deviation</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">Mean Absolute Error</div>
                  <div class="kpi-value">₹{metrics['MAE']:.2f}</div>
                  <div class="kpi-sub">Average Absolute Drift</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            mape_badge = "🎯 Elite Precision" if metrics['MAPE'] < 3.0 else "Good Precision"
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">Mean Abs % Error</div>
                  <div class="kpi-value">{metrics['MAPE']:.2f}%</div>
                  <div class="kpi-sub" style="color: #34d399; font-weight: 700;">{mape_badge}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m4:
            r2_badge = "🌟 High Correlation" if metrics['R2'] > 0.85 else "Moderate Fit"
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">R² Fit Score</div>
                  <div class="kpi-value">{metrics['R2']:.3f}</div>
                  <div class="kpi-sub" style="color: #38bdf8; font-weight: 700;">{r2_badge}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m5:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">Directional Accuracy</div>
                  <div class="kpi-value">{dir_acc:.1f}%</div>
                  <div class="kpi-sub">Trend Sign Concordance</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        st.markdown("### 📈 Actual vs LSTM Hold-Out Test Comparison")
        st.plotly_chart(
            line_compare(prepared["test_dates"], actual, predicted, "Hold-Out Validation Performance"),
            width="stretch",
        )

        st.markdown("### 🔍 Granular Prediction Journal")
        table = pd.DataFrame(
            {
                "Actual Close": actual,
                "Predicted Close": predicted,
                "Abs Error (INR)": np.abs(predicted - actual),
                "Error (%)": (np.abs(predicted - actual) / actual) * 100,
            },
            index=prepared["test_dates"],
        )
        st.dataframe(
            table.tail(15).style.format(
                {
                    "Actual Close": "₹{:.2f}",
                    "Predicted Close": "₹{:.2f}",
                    "Abs Error (INR)": "₹{:.2f}",
                    "Error (%)": "{:.2f}%",
                }
            ),
            width="stretch",
        )


# TAB 4: FORWARD FORECAST
with tab_forecast:
    if loaded is None:
        st.warning("Model bundle not found. Please train the model from the sidebar first.")
    else:
        model, scaler, meta = loaded
        last_window = prepared["scaled"][-lookback:]
        future_prices = forecast_next_days(model, last_window, prepared["scaler"], forecast_days)
        last_date = featured.index[-1]
        future_dates = pd.bdate_range(last_date + timedelta(days=1), periods=forecast_days)

        current_price = float(featured["Close"].iloc[-1])
        next_day_price = float(future_prices[0])
        next_day_change = next_day_price - current_price
        next_day_pct = (next_day_change / current_price) * 100

        end_price = float(future_prices[-1])
        total_horizon_change = end_price - current_price
        total_horizon_pct = (total_horizon_change / current_price) * 100

        trend_dir = "BULLISH" if total_horizon_change >= 0 else "BEARISH"
        trend_color = "#34d399" if total_horizon_change >= 0 else "#f87171"
        trend_icon = "▲" if total_horizon_change >= 0 else "▼"

        # Highlight card
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, rgba(19, 31, 56, 0.9) 0%, rgba(14, 24, 48, 0.95) 100%); border: 1px solid rgba(56, 189, 248, 0.35); border-radius: 16px; padding: 22px 26px; margin-bottom: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.4);">
              <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
                <div>
                  <div style="color: #7dd3fc; font-size: 0.8rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.06em;">Forward Forecast Signal</div>
                  <div style="font-size: 1.85rem; font-weight: 800; color: {trend_color}; margin-top: 4px;">
                    {trend_icon} {trend_dir} ({total_horizon_pct:+.2f}%)
                  </div>
                  <div style="color: #e2e8f0; font-size: 0.92rem; margin-top: 2px;">
                    Expected terminal price of <b>₹{end_price:,.2f}</b> by {future_dates[-1].strftime('%d %b %Y')}.
                  </div>
                </div>
                <div style="display: flex; gap: 26px; border-left: 1px solid rgba(255, 255, 255, 0.15); padding-left: 26px;">
                  <div>
                    <div style="color: #7dd3fc; font-size: 0.78rem; text-transform: uppercase; font-weight: 700;">Next Session Target</div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #ffffff; margin-top: 2px;">₹{next_day_price:,.2f}</div>
                    <div style="font-size: 0.84rem; font-weight: 700; color: {'#34d399' if next_day_change >= 0 else '#f87171'};">
                      {next_day_change:+.2f} ({next_day_pct:+.2f}%)
                    </div>
                  </div>
                  <div>
                    <div style="color: #7dd3fc; font-size: 0.78rem; text-transform: uppercase; font-weight: 700;">Current Close</div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #ffffff; margin-top: 2px;">₹{current_price:,.2f}</div>
                    <div style="font-size: 0.82rem; color: #94a3b8;">Baseline</div>
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        future_df = pd.DataFrame({"Forecast Close (INR)": future_prices}, index=future_dates)
        future_df.index.name = "Date"

        hist_tail = featured["Close"].tail(90)
        fig_fc = go.Figure()

        # Historical curve
        fig_fc.add_trace(
            go.Scatter(
                x=hist_tail.index,
                y=hist_tail.values,
                name="Historical Close",
                line=dict(color="#38bdf8", width=2),
            )
        )

        conn_dates = [hist_tail.index[-1]] + list(future_df.index)
        conn_prices = [hist_tail.values[-1]] + list(future_df["Forecast Close (INR)"])

        std_est = np.std(actual - predicted) if "actual" in locals() else current_price * 0.015
        upper_bound = [p + (i + 1) ** 0.5 * std_est * 0.8 for i, p in enumerate(conn_prices)]
        lower_bound = [p - (i + 1) ** 0.5 * std_est * 0.8 for i, p in enumerate(conn_prices)]

        fig_fc.add_trace(
            go.Scatter(
                x=conn_dates + conn_dates[::-1],
                y=upper_bound + lower_bound[::-1],
                fill="toself",
                fillcolor="rgba(16, 185, 129, 0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="Projected Corridor",
            )
        )

        fig_fc.add_trace(
            go.Scatter(
                x=conn_dates,
                y=conn_prices,
                name="LSTM Recursive Forecast",
                line=dict(color="#10b981", width=3, dash="solid"),
                marker=dict(size=6, color="#10b981"),
            )
        )

        fig_fc.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0c1426",
            height=480,
            font=dict(color="#e2e8f0"),
            yaxis=dict(title="Price (INR)", gridcolor="rgba(255, 255, 255, 0.08)"),
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.08)"),
            margin=dict(l=10, r=10, t=40, b=10),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(19, 31, 56, 0.8)",
                bordercolor="rgba(255, 255, 255, 0.15)",
                borderwidth=1,
                font=dict(size=11, color="#cbd5e1"),
            ),
        )
        st.plotly_chart(fig_fc, width="stretch")

        st.markdown("### 📅 Day-by-Day Forecast Schedule")
        day_names = [d.strftime("%A") for d in future_dates]
        changes_from_base = [p - current_price for p in future_prices]
        pcts_from_base = [(p - current_price) / current_price * 100 for p in future_prices]
        schedule_df = pd.DataFrame(
            {
                "Day": day_names,
                "Forecast Close": future_prices,
                "Expected Delta (INR)": changes_from_base,
                "Expected Delta (%)": pcts_from_base,
            },
            index=[d.strftime("%Y-%m-%d") for d in future_dates],
        )
        st.dataframe(
            schedule_df.style.format(
                {
                    "Forecast Close": "₹{:.2f}",
                    "Expected Delta (INR)": "₹{:+.2f}",
                    "Expected Delta (%)": "{:+.2f}%",
                }
            ),
            width="stretch",
        )

        st.markdown(
            """
            <div style="margin-top: 14px; font-size: 0.82rem; color: #94a3b8; line-height: 1.5;">
              ⚠ <b>Disclaimer:</b> Forecasted values are generated via recursive multi-step autoregression through an artificial neural network for research and analytical demonstration purposes only. Financial markets are subject to macroeconomic volatility and regime shifts. This does not constitute investment or trading advice.
            </div>
            """,
            unsafe_allow_html=True,
        )
