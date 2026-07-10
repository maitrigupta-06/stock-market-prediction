"""
Model architectures: Vanilla LSTM, GRU, and Bidirectional LSTM + Attention.
All three take the same windowed input shape and output a 3-vector
(High, Low, Close). Also includes Monte Carlo Dropout inference and
the High/Low/Close ordering constraint applied after prediction.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks


class BahdanauAttention(layers.Layer):
    """Additive attention over the sequence dimension, used by the BiLSTM model
    to learn which days in the look-back window matter most for the prediction."""

    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.W = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, hidden_states):
        score = self.V(tf.nn.tanh(self.W(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, tf.squeeze(weights, -1)


def build_vanilla_lstm(input_shape, n_outputs=3, units=64, dropout=0.2):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(units, return_sequences=True),
        layers.Dropout(dropout),
        layers.BatchNormalization(),
        layers.LSTM(units // 2),
        layers.Dropout(dropout),
        layers.Dense(32, activation='relu'),
        layers.Dense(n_outputs),
    ], name='VanillaLSTM')
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='huber', metrics=['mae'])
    return model


def build_gru(input_shape, n_outputs=3, units=64, dropout=0.2):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(units, return_sequences=True),
        layers.Dropout(dropout),
        layers.BatchNormalization(),
        layers.GRU(units // 2),
        layers.Dropout(dropout),
        layers.Dense(32, activation='relu'),
        layers.Dense(n_outputs),
    ], name='GRU')
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='huber', metrics=['mae'])
    return model


def build_bilstm_attention(input_shape, n_outputs=3, units=64, dropout=0.2):
    inp = layers.Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(units // 2, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)
    ctx, _ = BahdanauAttention(64)(x)
    x = layers.Dense(64, activation='relu')(ctx)
    x = layers.Dropout(dropout / 2)(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(n_outputs)(x)
    model = Model(inputs=inp, outputs=out, name='BiLSTM_Attention')
    model.compile(optimizer=keras.optimizers.Adam(5e-4), loss='huber', metrics=['mae'])
    return model


MODEL_BUILDERS = {
    'VanillaLSTM': build_vanilla_lstm,
    'GRU': build_gru,
    'BiLSTM_Attention': build_bilstm_attention,
}

COLORS = {
    'VanillaLSTM': '#4C72B0',
    'GRU': '#DD8452',
    'BiLSTM_Attention': '#55A868',
}


def get_callbacks():
    return [
        callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                 restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                     patience=7, min_lr=1e-6, verbose=1),
    ]


def mc_predict(model, X, n=50):
    """Runs the model n times with dropout active at inference (Monte Carlo
    Dropout) to produce a mean prediction and a 90% confidence interval."""
    preds = np.stack(
        [model(X, training=True).numpy() for _ in range(n)], axis=0
    )
    return preds.mean(0), np.percentile(preds, 5, axis=0), np.percentile(preds, 95, axis=0)


def enforce_hlc_order(arr, target_cols):
    """Reorders independently-predicted High/Low/Close so that
    High >= Close >= Low always holds, since nothing in the architecture
    itself enforces this relationship between the three outputs."""
    idx = {t: i for i, t in enumerate(target_cols)}
    h, l, c = arr[:, idx['High']], arr[:, idx['Low']], arr[:, idx['Close']]
    new_h = np.maximum.reduce([h, l, c])
    new_l = np.minimum.reduce([h, l, c])
    new_c = np.clip(c, new_l, new_h)
    out = arr.copy()
    out[:, idx['High']] = new_h
    out[:, idx['Low']] = new_l
    out[:, idx['Close']] = new_c
    return out
