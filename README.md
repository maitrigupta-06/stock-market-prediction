# Stock Market Prediction with Deep Learning

Predicting High, Low, and Close prices for any publicly traded stock using LSTM, GRU, and a Bidirectional LSTM with Attention — trained side by side, evaluated on held-out historical data, and used to forecast the next trading day with a Monte Carlo Dropout confidence interval.

## What this does

- Pulls live daily price history for any ticker directly from Yahoo Finance (works for US markets and Indian markets via `.NS` / `.BO` suffixes, priced in INR)
- Engineers 20 features from raw OHLCV data: SMA, EMA, MACD, Bollinger Bands, RSI, ATR, OBV, returns, and volatility
- Trains three architectures on the same 60-day look-back window and compares them on identical footing
- Predicts High, Low, and Close simultaneously per model, with output ordering enforced so `High >= Close >= Low` always holds
- Uses Monte Carlo Dropout (50 stochastic forward passes) to produce a 90% confidence interval around every prediction, instead of a single unqualified number
- Scores every model on RMSE, MAE, MAPE, R², and Directional Accuracy, then selects the best overall model via a normalized composite score across all three targets
- Forecasts the next trading day using the selected best model, with a plausibility check that flags predictions implying an unrealistic overnight price move
- Saves interactive Plotly charts: loss curves, predicted-vs-actual price with confidence bands, residuals, feature correlation, and a model comparison radar chart

## Project structure

```
stock_prediction/
├── __init__.py         package marker
├── data_pipeline.py     Yahoo Finance download, technical indicators, windowed train/test sequences
├── models.py             LSTM / GRU / BiLSTM+Attention architectures, attention layer, Monte Carlo Dropout, HLC ordering
├── evaluation.py         per-target metrics, best-model selection, next-day forecast, all Plotly charts
└── main.py                entry point: orchestrates the full pipeline end to end
```

## Getting started

```bash
pip install -r requirements.txt
python -m stock_prediction.main --ticker RELIANCE.NS
```

### Options

```
--ticker        Yahoo Finance symbol (default: INFY.NS)
                 US:  AAPL, MSFT, TSLA
                 NSE: INFY.NS, RELIANCE.NS, TCS.NS   -> INR
                 BSE: INFY.BO, RELIANCE.BO            -> INR
--start         Start date, YYYY-MM-DD (default: 2015-01-01)
--end           End date, YYYY-MM-DD (default: today)
--seq-len       Look-back window in trading days (default: 60)
--epochs        Max training epochs, early stopping applies (default: 100)
--batch-size    Training batch size (default: 32)
--mc-iter       Monte Carlo Dropout passes (default: 50)
--train-ratio   Train/test split ratio (default: 0.80)
--outdir        Directory to save HTML charts (default: plots)
```

Charts are saved as interactive `.html` files in `--outdir`; open them in any browser.

## Architecture comparison

| Model | Structure |
|---|---|
| Vanilla LSTM | 2-layer LSTM, dropout + batch norm |
| GRU | 2-layer GRU, dropout + batch norm |
| BiLSTM + Attention | Bidirectional LSTM stack with a Bahdanau attention layer over the sequence |

## Notes on the uncertainty estimates

The confidence intervals come from Monte Carlo Dropout: the model runs 50 times with dropout active at inference, and the spread of those outputs becomes the interval. This gives an honest sense of how confident the model actually is, rather than presenting a single number as if it were certain. A wide interval, or a next-day prediction that trips the plausibility check, is itself a meaningful result — it means the model doesn't have enough signal to make a confident call, which is common for single-day stock forecasts.

## Disclaimer

This project is for educational and portfolio purposes. It is not financial advice, and stock price prediction from historical patterns alone cannot account for news, earnings surprises, or other real-world events that materially move prices.
