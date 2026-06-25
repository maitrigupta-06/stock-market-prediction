"""
evaluation.py
Comprehensive metrics and Plotly-based visualisations.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    mape  = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    r2    = r2_score(y_true, y_pred)
    da    = directional_accuracy(y_true, y_pred)

    metrics = {"Model": label, "RMSE": rmse, "MAE": mae, "MAPE (%)": mape,
               "R²": r2, "Dir. Acc. (%)": da}
    return metrics


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage of time-steps where predicted direction matches actual."""
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return (true_dir == pred_dir).mean() * 100


def print_metrics_table(results: list[dict]) -> None:
    df = pd.DataFrame(results).set_index("Model")
    df = df.round(4)
    print("\n" + "=" * 65)
    print("MODEL COMPARISON — TEST SET METRICS")
    print("=" * 65)
    print(df.to_string())
    print("=" * 65 + "\n")


# ── Visualisations ────────────────────────────────────────────────────────────

COLORS = {
    "VanillaLSTM":      "#4C72B0",
    "GRU":              "#DD8452",
    "BiLSTM_Attention": "#55A868",
    "actual":           "#C44E52",
}


def plot_training_history(histories: dict, output_path: str = None) -> go.Figure:
    """Loss curves for all models on a single chart."""
    fig = make_subplots(rows=1, cols=1)

    for name, hist in histories.items():
        color = COLORS.get(name, "#999")
        epochs = list(range(1, len(hist["loss"]) + 1))
        fig.add_trace(go.Scatter(
            x=epochs, y=hist["loss"],
            mode="lines", name=f"{name} — train",
            line=dict(color=color, dash="solid"),
        ))
        if "val_loss" in hist:
            fig.add_trace(go.Scatter(
                x=epochs, y=hist["val_loss"],
                mode="lines", name=f"{name} — val",
                line=dict(color=color, dash="dash"),
            ))

    fig.update_layout(
        title="Training & Validation Loss (Huber)",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_white",
        height=450,
    )
    if output_path:
        fig.write_html(output_path)
    return fig


def plot_predictions(
    dates: np.ndarray,
    y_true: np.ndarray,
    predictions: dict,          # {model_name: y_pred_array}
    mc_bands: dict = None,      # {model_name: (mean, lower, upper)}
    ticker: str = "MSFT",
    output_path: str = None,
) -> go.Figure:
    """Actual vs predicted close price with optional MC uncertainty bands."""
    fig = go.Figure()

    # Actual prices
    fig.add_trace(go.Scatter(
        x=dates, y=y_true,
        mode="lines", name="Actual Close",
        line=dict(color=COLORS["actual"], width=2),
    ))

    for name, y_pred in predictions.items():
        color = COLORS.get(name, "#999")

        # MC uncertainty band
        if mc_bands and name in mc_bands:
            _, lo, hi = mc_bands[name]
            fig.add_trace(go.Scatter(
                x=np.concatenate([dates, dates[::-1]]),
                y=np.concatenate([hi, lo[::-1]]),
                fill="toself",
                fillcolor=color.replace(")", ", 0.15)").replace("rgb", "rgba") if "rgb" in color else color + "26",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{name} 90% CI",
                showlegend=True,
            ))

        fig.add_trace(go.Scatter(
            x=dates, y=y_pred,
            mode="lines", name=name,
            line=dict(color=color, width=1.5, dash="dot" if name != "BiLSTM_Attention" else "solid"),
        ))

    fig.update_layout(
        title=f"{ticker} — Predicted vs Actual Closing Price (Test Set)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if output_path:
        fig.write_html(output_path)
    return fig


def plot_residuals(
    dates: np.ndarray,
    y_true: np.ndarray,
    predictions: dict,
    output_path: str = None,
) -> go.Figure:
    """Residual (error) plot per model."""
    fig = go.Figure()
    for name, y_pred in predictions.items():
        color = COLORS.get(name, "#999")
        residuals = y_true - y_pred
        fig.add_trace(go.Scatter(
            x=dates, y=residuals,
            mode="lines", name=name,
            line=dict(color=color),
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4)
    fig.update_layout(
        title="Prediction Residuals (Actual − Predicted)",
        xaxis_title="Date",
        yaxis_title="Residual (USD)",
        template="plotly_white",
        height=380,
    )
    if output_path:
        fig.write_html(output_path)
    return fig


def plot_metrics_radar(results: list[dict], output_path: str = None) -> go.Figure:
    """Radar chart for at-a-glance model comparison."""
    metrics_to_plot = ["RMSE", "MAE", "MAPE (%)", "R²", "Dir. Acc. (%)"]
    df = pd.DataFrame(results).set_index("Model")

    # Normalise each metric to [0,1]; invert error metrics so higher = better
    norm = df[metrics_to_plot].copy()
    for col in ["RMSE", "MAE", "MAPE (%)"]:
        mx = norm[col].max()
        mn = norm[col].min()
        norm[col] = 1 - (norm[col] - mn) / (mx - mn + 1e-10)  # invert
    for col in ["R²", "Dir. Acc. (%)"]:
        mx = norm[col].max()
        mn = norm[col].min()
        norm[col] = (norm[col] - mn) / (mx - mn + 1e-10)

    fig = go.Figure()
    for name in norm.index:
        vals = norm.loc[name].tolist()
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=metrics_to_plot + [metrics_to_plot[0]],
            fill="toself",
            name=name,
            line=dict(color=COLORS.get(name, "#999")),
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Model Comparison Radar (higher = better on all axes)",
        template="plotly_white",
        height=480,
    )
    if output_path:
        fig.write_html(output_path)
    return fig


def plot_feature_correlation(raw_df: pd.DataFrame, feature_cols: list, output_path: str = None) -> go.Figure:
    """Heatmap of feature correlation with Close price."""
    corr = raw_df[feature_cols + ["Close"]].corr()[["Close"]].drop("Close")
    corr = corr.sort_values("Close", ascending=False)

    fig = go.Figure(go.Bar(
        x=corr["Close"],
        y=corr.index,
        orientation="h",
        marker=dict(
            color=corr["Close"],
            colorscale="RdBu",
            cmid=0,
        ),
    ))
    fig.update_layout(
        title="Feature Correlation with Close Price",
        xaxis_title="Pearson r",
        template="plotly_white",
        height=520,
    )
    if output_path:
        fig.write_html(output_path)
    return fig
