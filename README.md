# Stock Market Prediction with Deep Learning

Predict stock closing prices using three deep learning architectures trained on live market data with 20 technical indicators. Includes Monte Carlo Dropout uncertainty estimation and interactive Plotly visualisations.

---

## Project Overview

This notebook builds a complete end-to-end forecasting pipeline:

- **Live market data** fetched from Yahoo Finance for any ticker symbol
- **20 technical indicator features** — RSI, MACD, Bollinger Bands, ATR, OBV, moving averages and more
- **Three deep learning architectures** trained and benchmarked side by side
- **Monte Carlo Dropout** for prediction uncertainty (90% confidence intervals)
- **Five evaluation metrics** — RMSE, MAE, MAPE, R², Directional Accuracy
- **Interactive Plotly charts** for predictions, residuals, and model comparison

---

## Models

**Vanilla LSTM** — Two stacked LSTM layers with Dropout and BatchNormalization.

**GRU** — Gated Recurrent Unit network. Fewer parameters than LSTM, often trains faster with competitive accuracy.

**Bidirectional LSTM + Attention** — Reads the sequence both forwards and backwards. A Bahdanau attention layer then learns to focus on the most informative time-steps rather than relying solely on the final hidden state.

---

## Technical Indicators Used

| Group | Features |
|---|---|
| Price | Open, High, Low, Close |
| Volume | Volume, OBV (On-Balance Volume) |
| Trend | SMA 10, SMA 30, EMA 12, EMA 26 |
| Momentum | MACD, MACD Signal, MACD Histogram, RSI |
| Volatility | Bollinger Upper, Bollinger Lower, BB Width, ATR |
| Returns | Daily Return, 10-day Rolling Volatility |

---

## How to Run

### Google Colab (recommended)
Open the notebook directly in Colab and run all cells top to bottom. No setup needed beyond the first install cell.

### Local (Jupyter)
```bash
pip install tensorflow yfinance plotly scikit-learn pandas numpy
jupyter notebook stock_prediction.ipynb
```

---

## Configuration

At the top of the notebook, change these to whatever you want:

```python
TICKER     = 'MSFT'        # any Yahoo Finance ticker
START_DATE = '2015-01-01'
SEQ_LEN    = 60            # look-back window in trading days
EPOCHS     = 100
```

---

## Output Charts

| Chart | Description |
|---|---|
| Feature Correlation | Which indicators correlate most with the closing price |
| Training Loss | Train/validation loss curves for all three models |
| Predictions | Actual vs predicted price with 90% uncertainty bands |
| Residuals | Prediction error over time per model |
| Radar | At-a-glance model comparison across all five metrics |
