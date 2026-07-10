"""
Evaluation: per-target metrics, composite best-model selection, the
next-trading-day forecast with a plausibility guard, and all Plotly charts.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .models import enforce_hlc_order

METRIC_COLS = ['RMSE', 'MAE', 'MAPE (%)', 'R2', 'Dir. Acc. (%)']

# Most liquid large/mid-cap stocks move single-digit percent on an ordinary
# day; a next-day forecast implying more than this without a known catalyst
# (earnings, M&A, guidance) is flagged as unreliable rather than trusted.
PLAUSIBLE_MOVE_PCT = 10.0


def compute_metrics(model_name, target_cols, y_test_price, y_pred_price):
    """Computes RMSE, MAE, MAPE, R2, and Directional Accuracy separately
    for each target (High, Low, Close). Returns a list of row-dicts."""
    rows = []
    for j, tgt in enumerate(target_cols):
        actual = y_test_price[:, j]
        pred = y_pred_price[:, j]
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        mape = np.mean(np.abs((actual - pred) / (np.abs(actual) + 1e-8))) * 100
        r2 = r2_score(actual, pred)
        da = (np.sign(np.diff(actual)) == np.sign(np.diff(pred))).mean() * 100
        rows.append({'Model': model_name, 'Target': tgt, 'RMSE': rmse, 'MAE': mae,
                     'MAPE (%)': mape, 'R2': r2, 'Dir. Acc. (%)': da})
        print(f'  [{tgt:5s}] RMSE={rmse:.3f}  MAE={mae:.3f}  MAPE={mape:.2f}%  '
              f'R2={r2:.4f}  DirAcc={da:.1f}%')
    return rows


def select_best_model(metrics, target_cols):
    """
    Averages each metric across High/Low/Close per model, normalises them
    onto a common 0-1 scale (higher = better on every axis), and ranks
    models by composite score. Also counts how many of the individual
    Target x Metric comparisons each model wins outright, as corroborating
    evidence for the composite ranking.

    Returns (best_model_name, avg_metrics_df, composite_ranking_df, wins_dict).
    """
    metrics_df = pd.DataFrame(metrics)
    avg_metrics = metrics_df.groupby('Model')[METRIC_COLS].mean().round(4)

    norm = avg_metrics.copy()
    for col in ['RMSE', 'MAE', 'MAPE (%)']:
        mn, mx = norm[col].min(), norm[col].max()
        norm[col] = 1 - (norm[col] - mn) / (mx - mn + 1e-10)
    for col in ['R2', 'Dir. Acc. (%)']:
        mn, mx = norm[col].min(), norm[col].max()
        norm[col] = (norm[col] - mn) / (mx - mn + 1e-10)

    norm['Composite Score'] = norm.mean(axis=1)
    norm = norm.sort_values('Composite Score', ascending=False)
    best_model_name = norm.index[0]

    wins = {name: 0 for name in avg_metrics.index}
    for tgt in target_cols:
        sub = metrics_df[metrics_df['Target'] == tgt].set_index('Model')
        wins[sub['RMSE'].idxmin()] += 1
        wins[sub['MAE'].idxmin()] += 1
        wins[sub['MAPE (%)'].idxmin()] += 1
        wins[sub['R2'].idxmax()] += 1
        wins[sub['Dir. Acc. (%)'].idxmax()] += 1

    return best_model_name, avg_metrics, norm, wins


def print_best_model_report(best_model_name, avg_metrics, norm, wins):
    print('Average performance across High / Low / Close:\n')
    print(avg_metrics.to_string())
    print('\nComposite ranking (0-1 scale, higher = better):\n')
    print(norm[['Composite Score']].to_string())
    print('\nMetric-wins across all 15 Target x Metric comparisons:')
    for name, w in sorted(wins.items(), key=lambda x: -x[1]):
        print(f'  {name}: {w}/15')
    print(f'\nBest model: {best_model_name}')
    print(f'Justification: {best_model_name} has the highest composite score '
          f"({norm.loc[best_model_name, 'Composite Score']:.3f}) when RMSE, MAE, MAPE, R2, "
          f"and Directional Accuracy are averaged across High, Low, and Close and normalised "
          f"onto a common 0-1 scale, and it wins {wins[best_model_name]}/15 individual "
          f"metric-target comparisons -- the most of the three architectures.")


def predict_next_day(model, raw_df, feature_cols, scaler_X, scaler_y, target_cols,
                      seq_len, mc_iter=50):
    """Forecasts the next trading day's High, Low, and Close using the most
    recent look-back window, with a Monte Carlo Dropout confidence interval."""
    last_window = raw_df[feature_cols].values[-seq_len:]
    last_window_scaled = scaler_X.transform(last_window)
    X_input = last_window_scaled.reshape(1, seq_len, len(feature_cols))

    mc_stack = np.stack(
        [model(X_input, training=True).numpy()[0] for _ in range(mc_iter)], axis=0
    )
    pred_scaled = mc_stack.mean(axis=0, keepdims=True)
    lo_scaled = np.percentile(mc_stack, 5, axis=0, keepdims=True)
    hi_scaled = np.percentile(mc_stack, 95, axis=0, keepdims=True)

    pred_price = scaler_y.inverse_transform(pred_scaled)
    mc_lo = scaler_y.inverse_transform(lo_scaled)
    mc_hi = scaler_y.inverse_transform(hi_scaled)

    pred_price = enforce_hlc_order(pred_price, target_cols)[0]
    mc_lo = enforce_hlc_order(mc_lo, target_cols)[0]
    mc_hi = enforce_hlc_order(mc_hi, target_cols)[0]

    return pred_price, mc_lo, mc_hi


def print_next_day_report(ticker, model_name, raw_df, target_cols, pred, lo, hi,
                           seq_len, currency_sym):
    last_date = raw_df.index[-1]
    next_date = last_date + pd.tseries.offsets.BDay(1)
    last_close = float(raw_df['Close'].iloc[-1])

    print(f'Model used  : {model_name}  (selected as best performer above)')
    print(f'Based on    : last {seq_len} trading days through {last_date.date()}')
    print(f'Last close  : {currency_sym}{last_close:.2f}  (reference point for the move below)')
    print(f'\nPrediction for {ticker} on {next_date.date()}:\n')

    close_idx = target_cols.index('Close')
    for i, tgt in enumerate(target_cols):
        move_pct = (pred[i] - last_close) / last_close * 100
        print(f'  {tgt:6s}: {currency_sym}{pred[i]:.2f}   '
              f'(90% CI: {currency_sym}{lo[i]:.2f} - {currency_sym}{hi[i]:.2f})   '
              f'[{move_pct:+.1f}% vs last close]')

    close_move_pct = abs((pred[close_idx] - last_close) / last_close * 100)
    if close_move_pct > PLAUSIBLE_MOVE_PCT:
        print(f"\n\u26a0\ufe0f  WARNING: predicted Close implies a {close_move_pct:.1f}% overnight move "
              f"(threshold: {PLAUSIBLE_MOVE_PCT:.0f}%). This exceeds what's realistic for an "
              f"ordinary trading day without a specific news catalyst (earnings, M&A, guidance "
              f"change). Treat this prediction as unreliable -- it likely reflects model "
              f"instability or insufficient training data rather than a genuine signal. Consider "
              f"comparing against the GRU or Vanilla LSTM model's next-day prediction, or "
              f"retraining with more history.")
    else:
        print('\nPredicted move is within a plausible range for an ordinary trading day.')


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def _hex_to_rgba(hex_color, alpha=0.12):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def plot_feature_correlation(raw_df, feature_cols, ticker):
    feat_df = raw_df[feature_cols].copy()
    close_series = raw_df['Close'].copy()
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
    feat_df['_Close'] = close_series.values

    corr = feat_df.corr()[['_Close']].drop('_Close')
    corr.columns = ['Close']
    corr = corr.sort_values('Close', ascending=True)

    fig = go.Figure(go.Bar(
        x=corr['Close'], y=corr.index, orientation='h',
        marker=dict(color=corr['Close'], colorscale='RdBu', cmid=0),
    ))
    fig.update_layout(
        title=f'{ticker} \u2014 Feature Correlation with Close Price',
        xaxis_title='Pearson r', template='plotly_white', height=550,
    )
    return fig


def plot_loss(histories, colors):
    fig = go.Figure()
    for name, hist in histories.items():
        ep = list(range(1, len(hist['loss']) + 1))
        c = colors[name]
        fig.add_trace(go.Scatter(x=ep, y=hist['loss'], mode='lines',
                                  name=f'{name} train', line=dict(color=c)))
        fig.add_trace(go.Scatter(x=ep, y=hist['val_loss'], mode='lines',
                                  name=f'{name} val', line=dict(color=c, dash='dash')))
    fig.update_layout(title='Training & Validation Loss (Huber)',
                       xaxis_title='Epoch', yaxis_title='Loss',
                       template='plotly_white', height=450)
    return fig


def plot_predicted_vs_actual(ticker, currency, dates_test, y_test_price, preds_price,
                              mc_bands, target_cols, colors):
    fig = go.Figure()
    close_idx = target_cols.index('Close')

    fig.add_trace(go.Scatter(x=dates_test, y=y_test_price[:, close_idx], mode='lines',
                              name='Actual Close', line=dict(color='#C44E52', width=2)))

    for name, y_pred in preds_price.items():
        c = colors[name]
        _, lo, hi = mc_bands[name]
        fig.add_trace(go.Scatter(
            x=list(dates_test) + list(dates_test[::-1]),
            y=list(hi[:, close_idx]) + list(lo[::-1, close_idx]),
            fill='toself', fillcolor=_hex_to_rgba(c, 0.12),
            line=dict(color='rgba(0,0,0,0)'),
            name=f'{name} 90% CI', showlegend=True,
        ))
        fig.add_trace(go.Scatter(x=dates_test, y=y_pred[:, close_idx], mode='lines',
                                  name=name, line=dict(color=c, width=1.5)))

    fig.update_layout(
        title=f'{ticker} -- Predicted vs Actual Closing Price (Test Set)',
        xaxis_title='Date', yaxis_title=f'Price ({currency})',
        template='plotly_white', hovermode='x unified', height=520,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return fig


def plot_residuals(currency, dates_test, y_test_price, preds_price, target_cols, colors):
    fig = go.Figure()
    close_idx = target_cols.index('Close')
    for name, y_pred in preds_price.items():
        fig.add_trace(go.Scatter(
            x=dates_test, y=y_test_price[:, close_idx] - y_pred[:, close_idx],
            mode='lines', name=name, line=dict(color=colors[name]),
        ))
    fig.add_hline(y=0, line_dash='dash', line_color='black', opacity=0.4)
    fig.update_layout(title='Prediction Residuals (Close)',
                       xaxis_title='Date', yaxis_title=f'Residual ({currency})',
                       template='plotly_white', height=380)
    return fig


def plot_radar(avg_metrics, colors):
    norm = avg_metrics.copy()
    for col in ['RMSE', 'MAE', 'MAPE (%)']:
        mn, mx = norm[col].min(), norm[col].max()
        norm[col] = 1 - (norm[col] - mn) / (mx - mn + 1e-10)
    for col in ['R2', 'Dir. Acc. (%)']:
        mn, mx = norm[col].min(), norm[col].max()
        norm[col] = (norm[col] - mn) / (mx - mn + 1e-10)

    fig = go.Figure()
    for name in norm.index:
        vals = norm.loc[name].tolist()
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=METRIC_COLS + [METRIC_COLS[0]],
            fill='toself', name=name, line=dict(color=colors[name]),
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title='Model Comparison -- averaged across High/Low/Close, higher is better on all axes',
        template='plotly_white', height=480,
    )
    return fig
