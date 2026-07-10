"""
Data pipeline: pulls historical OHLCV data from Yahoo Finance, engineers
technical indicator features, and builds windowed train/test sequences
for the High, Low, and Close targets.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_10', 'SMA_30', 'EMA_12', 'EMA_26',
    'MACD', 'MACD_signal', 'MACD_hist',
    'BB_upper', 'BB_lower', 'BB_width',
    'RSI', 'ATR', 'OBV', 'Return', 'Volatility',
]

TARGET_COLS = ['High', 'Low', 'Close']


def infer_currency(ticker):
    """NSE/BSE tickers (.NS / .BO) are already priced in INR by Yahoo Finance."""
    if ticker.upper().endswith(('.NS', '.BO')):
        return 'INR', '\u20b9'
    return 'USD', '$'


def add_technical_indicators(df):
    """Adds 15 technical indicator columns derived from OHLCV price data."""
    close = df['Close']

    # Moving averages
    df['SMA_10'] = close.rolling(10).mean()
    df['SMA_30'] = close.rolling(30).mean()
    df['EMA_12'] = close.ewm(span=12, adjust=False).mean()
    df['EMA_26'] = close.ewm(span=26, adjust=False).mean()

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['BB_upper'] = bb_mid + 2 * bb_std
    df['BB_lower'] = bb_mid - 2 * bb_std
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / bb_mid

    # RSI (14-day)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # ATR
    hl = df['High'] - df['Low']
    hpc = (df['High'] - close.shift(1)).abs()
    lpc = (df['Low'] - close.shift(1)).abs()
    df['ATR'] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()

    # On-Balance Volume
    df['OBV'] = (np.sign(close.diff()) * df['Volume']).fillna(0).cumsum()

    # Return & volatility
    df['Return'] = close.pct_change()
    df['Volatility'] = df['Return'].rolling(10).std()

    return df


def prepare_data(ticker, start, end, seq_len, train_ratio):
    """
    Downloads price history, engineers features, and builds windowed
    sequences for training. Targets are High, Low, and Close simultaneously.

    Returns a dict with train/test arrays, fitted scalers, the enriched
    dataframe, and the feature/target column lists.
    """
    print(f'Downloading {ticker} data...')
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    raw = raw.loc[:, ~raw.columns.duplicated()]
    raw.dropna(inplace=True)

    df = add_technical_indicators(raw.copy())
    df.dropna(inplace=True)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(df[FEATURE_COLS])
    y_scaled = scaler_y.fit_transform(df[TARGET_COLS])

    X_seq, y_seq, dates = [], [], []
    for i in range(seq_len, len(df)):
        X_seq.append(X_scaled[i - seq_len: i])
        y_seq.append(y_scaled[i])
        dates.append(df.index[i])

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)   # shape (N, 3) -> High, Low, Close
    dates = np.array(dates)

    split = int(len(X_seq) * train_ratio)
    return {
        'X_train': X_seq[:split], 'X_test': X_seq[split:],
        'y_train': y_seq[:split], 'y_test': y_seq[split:],
        'dates_train': dates[:split], 'dates_test': dates[split:],
        'scaler_X': scaler_X, 'scaler_y': scaler_y, 'raw_df': df,
        'feature_cols': FEATURE_COLS, 'target_cols': TARGET_COLS,
    }
