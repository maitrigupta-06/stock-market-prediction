"""
main.py
─────────────────────────────────────────────────────────────────────────────
Stock Market Prediction with Deep Learning
─────────────────────────────────────────────────────────────────────────────
End-to-end pipeline for forecasting stock closing prices using three deep
learning architectures trained on live market data with 20 technical
indicator features.

  · Live data via yfinance (any ticker, any date range)
  · 20 technical indicators: RSI, MACD, Bollinger Bands, ATR, OBV …
  · Three models: Vanilla LSTM | GRU | BiLSTM + Attention
  · Monte Carlo Dropout for prediction uncertainty (90% CI)
  · Five metrics: RMSE, MAE, MAPE, R², Directional Accuracy
  · Interactive Plotly charts saved as standalone HTML files
  · Early stopping + ReduceLROnPlateau for every model

Usage:
    python main.py                          # defaults: MSFT, 2015–today
    python main.py --ticker AAPL --epochs 80
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

# ── path setup so sub-packages are importable ────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from data.data_pipeline import fetch_and_prepare
from models.models import (
    build_vanilla_lstm,
    build_gru,
    build_bilstm_attention,
    get_callbacks,
    monte_carlo_predict,
)
from evaluation.evaluation import (
    compute_metrics,
    print_metrics_table,
    plot_training_history,
    plot_predictions,
    plot_residuals,
    plot_metrics_radar,
    plot_feature_correlation,
)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Enhanced Stock Market Predictor")
    p.add_argument("--ticker",   default="MSFT", help="Yahoo Finance ticker (default: MSFT)")
    p.add_argument("--start",    default="2015-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end",      default=None,  help="End date (YYYY-MM-DD), default: today")
    p.add_argument("--seq_len",  default=60, type=int, help="Look-back window (default: 60 days)")
    p.add_argument("--epochs",   default=100, type=int, help="Max training epochs per model")
    p.add_argument("--batch",    default=32,  type=int, help="Batch size")
    p.add_argument("--mc_iter",  default=50,  type=int, help="Monte Carlo dropout iterations")
    p.add_argument("--out_dir",  default="outputs", help="Directory to save HTML charts")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    data = fetch_and_prepare(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        sequence_length=args.seq_len,
    )

    X_train, X_test   = data["X_train"], data["X_test"]
    y_train, y_test   = data["y_train"], data["y_test"]
    scaler_y          = data["scaler_y"]
    dates_test        = data["dates_test"]
    ticker            = data["ticker"]

    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"\n[info] Input shape  : {input_shape}")
    print(f"[info] Train samples: {len(X_train)}")
    print(f"[info] Test  samples: {len(X_test)}\n")

    # Inverse-scale y for dollar-value metrics
    y_test_price  = scaler_y.inverse_transform(y_test.reshape(-1, 1)).squeeze()

    # ── 2. Models ─────────────────────────────────────────────────────────────
    model_builders = {
        "VanillaLSTM":      lambda: build_vanilla_lstm(input_shape),
        "GRU":              lambda: build_gru(input_shape),
        "BiLSTM_Attention": lambda: build_bilstm_attention(input_shape),
    }

    trained_models = {}
    histories      = {}
    predictions    = {}
    mc_bands       = {}
    metrics_list   = []

    for name, builder in model_builders.items():
        print(f"\n{'='*60}")
        print(f"  Training: {name}")
        print(f"{'='*60}")
        t0 = time.time()

        model = builder()
        model.summary()

        hist = model.fit(
            X_train, y_train,
            epochs=args.epochs,
            batch_size=args.batch,
            validation_split=0.1,
            callbacks=get_callbacks(),
            verbose=1,
            shuffle=False,     # time-series — keep order
        )

        elapsed = time.time() - t0
        print(f"[{name}] Training time: {elapsed:.1f}s")

        trained_models[name] = model
        histories[name]      = hist.history

        # Point predictions (scaled → price)
        y_pred_scaled = model.predict(X_test, verbose=0).squeeze()
        y_pred_price  = scaler_y.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).squeeze()
        predictions[name] = y_pred_price

        # Monte Carlo uncertainty
        mc_mean_s, mc_lo_s, mc_hi_s = monte_carlo_predict(model, X_test, args.mc_iter)
        mc_bands[name] = (
            scaler_y.inverse_transform(mc_mean_s.reshape(-1, 1)).squeeze(),
            scaler_y.inverse_transform(mc_lo_s.reshape(-1, 1)).squeeze(),
            scaler_y.inverse_transform(mc_hi_s.reshape(-1, 1)).squeeze(),
        )

        # Metrics
        m = compute_metrics(y_test_price, y_pred_price, label=name)
        metrics_list.append(m)
        print(f"[{name}] RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
              f"MAPE={m['MAPE (%)']:.2f}%  R²={m['R²']:.4f}  "
              f"DirAcc={m['Dir. Acc. (%)']:.1f}%")

    # ── 3. Results ────────────────────────────────────────────────────────────
    print_metrics_table(metrics_list)

    # Save metrics CSV
    pd.DataFrame(metrics_list).to_csv(
        os.path.join(args.out_dir, "metrics.csv"), index=False
    )
    print(f"[info] Metrics saved → {args.out_dir}/metrics.csv")

    # ── 4. Visualisations ─────────────────────────────────────────────────────
    charts = {
        "training_loss.html":        lambda p: plot_training_history(histories, p),
        "predictions.html":          lambda p: plot_predictions(
                                        dates_test, y_test_price,
                                        predictions, mc_bands, ticker, p),
        "residuals.html":            lambda p: plot_residuals(
                                        dates_test, y_test_price, predictions, p),
        "model_radar.html":          lambda p: plot_metrics_radar(metrics_list, p),
        "feature_correlation.html":  lambda p: plot_feature_correlation(
                                        data["raw_df"], data["feature_names"], p),
    }

    for filename, fn in charts.items():
        path = os.path.join(args.out_dir, filename)
        fn(path)
        print(f"[chart] Saved → {path}")

    print(f"\n✅  All done! Open the HTML files in {args.out_dir}/ for interactive charts.\n")


if __name__ == "__main__":
    main()
