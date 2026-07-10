"""
Entry point. Downloads data, trains all three models, evaluates them,
selects the best overall model, forecasts the next trading day, and
saves all charts as interactive HTML files.

Usage:
    python -m stock_prediction.main
    python -m stock_prediction.main --ticker RELIANCE.NS --epochs 50
"""

import argparse
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import tensorflow as tf

from .data_pipeline import prepare_data, infer_currency
from .models import MODEL_BUILDERS, COLORS, get_callbacks, mc_predict, enforce_hlc_order
from .evaluation import (
    compute_metrics, select_best_model, print_best_model_report,
    predict_next_day, print_next_day_report,
    plot_feature_correlation, plot_loss, plot_predicted_vs_actual,
    plot_residuals, plot_radar,
)


def parse_args():
    p = argparse.ArgumentParser(description='Stock price prediction with LSTM/GRU/BiLSTM-Attention')
    p.add_argument('--ticker', default='INFY.NS',
                    help="Yahoo Finance ticker. US: 'AAPL'. NSE (India): 'INFY.NS', "
                         "'RELIANCE.NS'. BSE (India): 'INFY.BO'.")
    p.add_argument('--start', default='2015-01-01', help='Start date (YYYY-MM-DD)')
    p.add_argument('--end', default=None, help='End date (YYYY-MM-DD); default = today')
    p.add_argument('--seq-len', type=int, default=60, help='Look-back window in trading days')
    p.add_argument('--epochs', type=int, default=100, help='Max training epochs (early stopping applies)')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--mc-iter', type=int, default=50, help='Monte Carlo Dropout passes')
    p.add_argument('--train-ratio', type=float, default=0.80)
    p.add_argument('--outdir', default='plots', help='Directory to save HTML charts')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    currency, currency_sym = infer_currency(args.ticker)
    print(f'Ticker: {args.ticker}  ->  currency: {currency} ({currency_sym})')

    # -- Data --------------------------------------------------------------
    data = prepare_data(args.ticker, args.start, args.end, args.seq_len, args.train_ratio)
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    scaler_X, scaler_y = data['scaler_X'], data['scaler_y']
    target_cols = data['target_cols']
    dates_test = data['dates_test']
    input_shape = (X_train.shape[1], X_train.shape[2])
    y_test_price = scaler_y.inverse_transform(y_test)

    print(f'Rows after dropna: {len(data["raw_df"])}')
    print(f'Train sequences  : {X_train.shape}')
    print(f'Test sequences   : {X_test.shape}')

    fig = plot_feature_correlation(data['raw_df'], data['feature_cols'], args.ticker)
    fig.write_html(os.path.join(args.outdir, 'feature_correlation.html'))

    # -- Train all three models ---------------------------------------------
    trained, histories, preds_price, mc_bands, metrics = {}, {}, {}, {}, []

    for name, builder in MODEL_BUILDERS.items():
        print(f"\n{'=' * 55}\n  {name}\n{'=' * 55}")
        model = builder(input_shape, n_outputs=len(target_cols))
        hist = model.fit(
            X_train, y_train,
            epochs=args.epochs, batch_size=args.batch_size,
            validation_split=0.1, callbacks=get_callbacks(),
            verbose=1, shuffle=False,
        )
        trained[name] = model
        histories[name] = hist.history

        # Point prediction = MC mean, so it is consistent with the CI
        mc_m, mc_lo, mc_hi = mc_predict(model, X_test, args.mc_iter)
        y_pred_p = enforce_hlc_order(scaler_y.inverse_transform(mc_m), target_cols)
        mc_lo_p = enforce_hlc_order(scaler_y.inverse_transform(mc_lo), target_cols)
        mc_hi_p = enforce_hlc_order(scaler_y.inverse_transform(mc_hi), target_cols)

        preds_price[name] = y_pred_p
        mc_bands[name] = (y_pred_p, mc_lo_p, mc_hi_p)

        metrics.extend(compute_metrics(name, target_cols, y_test_price, y_pred_p))

    # -- Results table -------------------------------------------------------
    results_df = pd.DataFrame(metrics).set_index(['Model', 'Target']).round(4)
    print('\n' + results_df.to_string())

    # -- Charts ----------------------------------------------------------------
    plot_loss(histories, COLORS).write_html(os.path.join(args.outdir, 'loss.html'))
    plot_predicted_vs_actual(
        args.ticker, currency, dates_test, y_test_price, preds_price, mc_bands, target_cols, COLORS
    ).write_html(os.path.join(args.outdir, 'predicted_vs_actual.html'))
    plot_residuals(
        currency, dates_test, y_test_price, preds_price, target_cols, COLORS
    ).write_html(os.path.join(args.outdir, 'residuals.html'))

    # -- Best model selection -----------------------------------------------
    best_model_name, avg_metrics, norm, wins = select_best_model(metrics, target_cols)
    print()
    print_best_model_report(best_model_name, avg_metrics, norm, wins)

    plot_radar(avg_metrics, COLORS).write_html(os.path.join(args.outdir, 'radar.html'))

    # -- Next trading day forecast -------------------------------------------
    best_model = trained[best_model_name]
    pred, lo, hi = predict_next_day(
        best_model, data['raw_df'], data['feature_cols'], scaler_X, scaler_y,
        target_cols, args.seq_len, args.mc_iter,
    )
    print()
    print_next_day_report(
        args.ticker, best_model_name, data['raw_df'], target_cols, pred, lo, hi,
        args.seq_len, currency_sym,
    )

    print(f'\nCharts saved to: {os.path.abspath(args.outdir)}/')


if __name__ == '__main__':
    main()
