"""
models.py
Three model architectures for comparison:
  1. Vanilla LSTM  (baseline — what the original article used)
  2. GRU           (lighter, often competitive)
  3. Bidirectional LSTM + Self-Attention  (most expressive)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


# ── Attention Layer ──────────────────────────────────────────────────────────

class BahdanauAttention(layers.Layer):
    """Additive attention; produces a context vector from an LSTM sequence."""

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.W = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, hidden_states):
        # hidden_states: (batch, timesteps, features)
        score = self.V(tf.nn.tanh(self.W(hidden_states)))   # (batch, T, 1)
        weights = tf.nn.softmax(score, axis=1)               # (batch, T, 1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)  # (batch, features)
        return context, tf.squeeze(weights, -1)              # also return weights


# ── Model Builders ───────────────────────────────────────────────────────────

def build_vanilla_lstm(
    input_shape: tuple,
    lstm_units: int = 64,
    dropout: float = 0.2,
) -> keras.Model:
    """
    Simple stacked LSTM — improved version of the article's baseline.
    Two LSTM layers, dropout regularisation, batch normalisation.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(lstm_units, return_sequences=True),
        layers.Dropout(dropout),
        layers.BatchNormalization(),
        layers.LSTM(lstm_units // 2, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ], name="VanillaLSTM")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="huber",          # more robust to outliers than MSE
        metrics=["mae"],
    )
    return model


def build_gru(
    input_shape: tuple,
    gru_units: int = 64,
    dropout: float = 0.2,
) -> keras.Model:
    """Stacked GRU — fewer parameters, often trains faster than LSTM."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(gru_units, return_sequences=True),
        layers.Dropout(dropout),
        layers.BatchNormalization(),
        layers.GRU(gru_units // 2, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ], name="GRU")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="huber",
        metrics=["mae"],
    )
    return model


def build_bilstm_attention(
    input_shape: tuple,
    lstm_units: int = 64,
    dropout: float = 0.2,
    attention_units: int = 64,
) -> keras.Model:
    """
    Bidirectional LSTM + Bahdanau self-attention.
    The attention mechanism lets the model focus on the most informative
    time-steps rather than relying solely on the final hidden state.
    """
    inp = layers.Input(shape=input_shape)

    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True)
    )(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Bidirectional(
        layers.LSTM(lstm_units // 2, return_sequences=True)
    )(x)
    x = layers.Dropout(dropout)(x)

    # Attention
    context, _ = BahdanauAttention(attention_units)(x)

    x = layers.Dense(64, activation="relu")(context)
    x = layers.Dropout(dropout / 2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)

    model = Model(inputs=inp, outputs=out, name="BiLSTM_Attention")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="huber",
        metrics=["mae"],
    )
    return model


# ── Common Training Helper ───────────────────────────────────────────────────

def get_callbacks(monitor: str = "val_loss", patience: int = 15) -> list:
    return [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def monte_carlo_predict(model, X, n_iter: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run forward passes with dropout active (MC Dropout) to estimate
    prediction uncertainty.

    Returns:
        mean, lower_bound (5th pct), upper_bound (95th pct)
    """
    predictions = np.stack(
        [model(X, training=True).numpy().squeeze() for _ in range(n_iter)],
        axis=0
    )  # (n_iter, n_samples)
    mean = predictions.mean(axis=0)
    lower = np.percentile(predictions, 5, axis=0)
    upper = np.percentile(predictions, 95, axis=0)
    return mean, lower, upper
