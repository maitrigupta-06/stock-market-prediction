"""
data_pipeline.py
Fetches stock data via yfinance and engineers technical indicator features.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


# ── Technical Indicators ────────────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RSI, MACD, Bollinger Bands, and moving averages.
    All computed from scratch to avoid TA-lib dependency issues.
    """
    close = df["Close"]

    # Simple & Exponential Moving Averages
    df["SMA_10"] = close.rolling(10).mean()
    df["SMA_30"] = close.rolling(30).mean()
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # MACD and Signal line
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Bollinger Bands (20-day, 2σ)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / bb_mid

    # RSI (14-day)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Average True Range (volatility proxy)
    high_low = df["High"] - df["Low"]
    high_pc = (df["High"] - close.shift(1)).abs()
    low_pc = (df["Low"] - close.shift(1)).abs()
    df["ATR"] = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1).rolling(14).mean()

    # On-Balance Volume
    obv = (np.sign(close.diff()) * df["Volume"]).fillna(0).cumsum()
    df["OBV"] = obv

    # Daily return and rolling volatility
    df["Return"] = close.pct_change()
    df["Volatility"] = df["Return"].rolling(10).std()

    return df


# ── Data Fetching & Preparation ─────────────────────────────────────────────

def fetch_and_prepare(
    ticker: str = "MSFT",
    start: str = "2010-01-01",
    end: str = None,
    sequence_length: int = 60,
) -> dict:
    """
    Downloads OHLCV data, engineers features, scales, and creates
    sliding-window sequences ready for LSTM input.

    Returns a dict with keys:
        X_train, X_test, y_train, y_test,
        scaler_y, dates_train, dates_test,
        feature_names, raw_df
    """
    print(f"[data] Downloading {ticker} …")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    raw.columns = raw.columns.droplevel(1) if isinstance(raw.columns, pd.MultiIndex) else raw.columns
    raw.dropna(inplace=True)

    df = raw.copy()
    df = add_technical_indicators(df)
    df.dropna(inplace=True)          # drop NaN rows from rolling windows
    df.sort_index(inplace=True)

    # Target: next-day Adj Close (we use "Close" since auto_adjust=True)
    target_col = "Close"

    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_10", "SMA_30", "EMA_12", "EMA_26",
        "MACD", "MACD_signal", "MACD_hist",
        "BB_upper", "BB_lower", "BB_width",
        "RSI", "ATR", "OBV", "Return", "Volatility",
    ]

    # Scale features and target separately
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(df[feature_cols])
    y_scaled = scaler_y.fit_transform(df[[target_col]])

    # Build sliding windows
    X_seq, y_seq, dates_seq = [], [], []
    for i in range(sequence_length, len(df)):
        X_seq.append(X_scaled[i - sequence_length: i])
        y_seq.append(y_scaled[i, 0])
        dates_seq.append(df.index[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    dates_seq = np.array(dates_seq)

    # 80 / 20 chronological split
    split = int(len(X_seq) * 0.80)
    return {
        "X_train": X_seq[:split],
        "X_test":  X_seq[split:],
        "y_train": y_seq[:split],
        "y_test":  y_seq[split:],
        "scaler_y": scaler_y,
        "dates_train": dates_seq[:split],
        "dates_test":  dates_seq[split:],
        "feature_names": feature_cols,
        "raw_df": df,
        "ticker": ticker,
        "sequence_length": sequence_length,
    }
