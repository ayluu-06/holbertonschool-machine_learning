#!/usr/bin/env python3
"""
modulo documentado
"""
import argparse
import os
from typing import Tuple

import numpy as np
import tensorflow as tf


def make_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    """
    funcion documentada
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        buffer_size = min(int(X.shape[0]), 10000)
        ds = ds.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(
    timesteps: int,
    features: int,
    units: int,
    dropout: float,
) -> tf.keras.Model:
    """
    funcion documentada
    """
    inputs = tf.keras.Input(shape=(timesteps, features))
    x = tf.keras.layers.LSTM(units, return_sequences=True)(inputs)
    if dropout > 0.0:
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LSTM(units)(x)
    if dropout > 0.0:
        x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(), loss="mse", metrics=["mse"])
    return model


def parse_args() -> argparse.Namespace:
    """
    funcion documentada
    """
    parser = argparse.ArgumentParser(
        description="Train/validate BTC forecasting RNN")
    parser.add_argument(
        "--data", type=str,
        default="processed_btc.npz", help="Path to preprocessed .npz")
    parser.add_argument(
        "--model", type=str, default="btc_rnn.keras", help="Output model path")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Max epochs")
    parser.add_argument(
        "--units", type=int, default=64, help="LSTM units")
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate")
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()


def load_npz(path: str) -> Tuple[np.ndarray,
                                 np.ndarray, np.ndarray, np.ndarray]:
    """
    funcion documentada
    """
    data = np.load(path, allow_pickle=True)
    return data["X_train"], data["y_train"], data["X_val"], data["y_val"]


def main() -> None:
    """
    funcion documentada
    """
    args = parse_args()
    tf.keras.utils.set_random_seed(int(args.seed))

    X_train, y_train, X_val, y_val = load_npz(args.data)

    timesteps = int(X_train.shape[1])
    features = int(X_train.shape[2])

    train_ds = make_dataset(X_train, y_train, args.batch_size, shuffle=True)
    val_ds = make_dataset(X_val, y_val, args.batch_size, shuffle=False)

    model = build_model(
        timesteps=timesteps,
        features=features,
        units=int(args.units),
        dropout=float(args.dropout),
    )

    out_dir = os.path.dirname(args.model)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(args.patience),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.model,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(args.epochs),
        callbacks=callbacks,
        verbose=1,
    )

    model.save(args.model)


if __name__ == "__main__":
    main()
