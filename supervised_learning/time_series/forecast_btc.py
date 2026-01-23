#!/usr/bin/env python3
"""
modulo documentado
"""
import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    funcion documentada
    """
    rename_map = {
        "Timestamp": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume_(BTC)": "vol_btc",
        "Volume_(Currency)": "vol_usd",
        "Weighted_Price": "vwap",
        "Volume BTC": "vol_btc",
        "Volume USD": "vol_usd",
        "Weighted Price": "vwap",
    }
    cols = {}
    for c in df.columns:
        if c in rename_map:
            cols[c] = rename_map[c]
    df = df.rename(columns=cols)

    if "timestamp" not in df.columns:
        raise ValueError("Dataset must include a Timestamp column")

    keep = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "vol_btc",
        "vol_usd",
        "vwap",
    ]
    present = [c for c in keep if c in df.columns]
    return df[present]


def _load_csv(path: str) -> pd.DataFrame:
    """
    funcion documentada
    """
    df = pd.read_csv(path)
    df = _standardize_columns(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")
    return df


def _resample_minutely(df: pd.DataFrame) -> pd.DataFrame:
    """
    funcion documentada
    """
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="1min",
        tz=df.index.tz,
    )
    df = df.reindex(full_index)

    numeric_cols = df.columns.tolist()
    df[numeric_cols] = df[numeric_cols].astype("float64")

    df = df.ffill()
    df = df.dropna()
    return df


def _combine_sources(
    coinbase_df: Optional[pd.DataFrame],
    bitstamp_df: Optional[pd.DataFrame],
    source: str,
) -> pd.DataFrame:
    """
    funcion documentada
    """
    source = source.lower().strip()
    if source not in {"coinbase", "bitstamp", "both"}:
        raise ValueError("--source must be one of: coinbase, bitstamp, both")

    if source == "coinbase":
        if coinbase_df is None:
            raise ValueError("coinbase path required for --source coinbase")
        return _resample_minutely(coinbase_df)

    if source == "bitstamp":
        if bitstamp_df is None:
            raise ValueError("bitstamp path required for --source bitstamp")
        return _resample_minutely(bitstamp_df)

    if coinbase_df is None or bitstamp_df is None:
        raise ValueError(
            "Both coinbase and bitstamp paths required for --source both"
            )

    cb = _resample_minutely(coinbase_df)
    bs = _resample_minutely(bitstamp_df)

    merged = cb.join(bs, how="inner", lsuffix="_cb", rsuffix="_bs")
    if merged.empty:
        raise ValueError("No overlapping timestamps between datasets")

    base_cols = ["open", "high", "low", "close", "vol_btc", "vol_usd", "vwap"]
    out: Dict[str, np.ndarray] = {}
    for c in base_cols:
        c_cb = f"{c}_cb"
        c_bs = f"{c}_bs"
        if c_cb in merged.columns and c_bs in merged.columns:
            out[c] = np.nanmean(
                np.stack([merged[c_cb].to_numpy(),
                          merged[c_bs].to_numpy()], axis=0),
                axis=0,
            )
        elif c_cb in merged.columns:
            out[c] = merged[c_cb].to_numpy()
        elif c_bs in merged.columns:
            out[c] = merged[c_bs].to_numpy()

    out_df = pd.DataFrame(out, index=merged.index)
    out_df = out_df.ffill().dropna()
    return out_df


def _select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    funcion documentada
    """
    preferred = ["close", "vol_btc", "vol_usd", "vwap", "high", "low", "open"]
    feats = [c for c in preferred if c in df.columns]

    if "close" not in feats:
        raise ValueError("A 'close' feature is required")

    df = df[feats].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df, feats


def _build_sequences(
    values: np.ndarray,
    close_index: int,
    seq_len: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    funcion documentada
    """
    n = values.shape[0]
    last_start = n - seq_len - horizon
    if last_start <= 0:
        raise ValueError(
            "Not enough data to build sequences with given settings"
            )

    xs: List[np.ndarray] = []
    ys: List[float] = []

    for start in range(last_start):
        end = start + seq_len
        target_t = end + horizon - 1
        xs.append(values[start:end])
        ys.append(values[target_t, close_index])

    X = np.stack(xs, axis=0).astype("float32")
    y = np.array(ys, dtype="float32").reshape(-1, 1)
    return X, y


def _time_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    funcion documentada
    """
    n = X.shape[0]
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train = X[:train_end]
    y_train = y[:train_end]
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    X_test = X[val_end:]
    y_test = y[val_end:]
    return X_train, y_train, X_val, y_val, X_test, y_test


def _fit_standardizer(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    funcion documentada
    """
    flat = X_train.reshape(-1, X_train.shape[-1]).astype("float64")
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype("float32"), std.astype("float32")


def _apply_standardizer(X: np.ndarray,
                        mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    funcion documentada
    """
    return ((X - mean) / std).astype("float32")


def parse_args() -> argparse.Namespace:
    """
    funcion documentada
    """
    parser = argparse.ArgumentParser(
        description="Preprocess BTC datasets for RNNs"
        )
    parser.add_argument(
        "--coinbase", type=str, default=None, help="Path to coinbase CSV"
        )
    parser.add_argument(
        "--bitstamp", type=str, default=None, help="Path to bitstamp CSV"
        )
    parser.add_argument(
        "--source",
        type=str,
        default="bitstamp",
        choices=["coinbase", "bitstamp", "both"],
        help="Which dataset(s) to use",
    )
    parser.add_argument(
        "--out", type=str, default="processed_btc.npz", help="Output .npz path"
        )
    parser.add_argument(
        "--seq-len", type=int, default=24 * 60,
        help="Minutes of history (default 1440)"
        )
    parser.add_argument(
        "--horizon", type=int, default=60,
        help="Minutes ahead to predict (default 60)"
        )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Train split ratio"
        )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Validation split ratio"
        )
    return parser.parse_args()


def main() -> None:
    """
    funcion documentada
    """
    args = parse_args()

    coinbase_df = _load_csv(args.coinbase) if args.coinbase else None
    bitstamp_df = _load_csv(args.bitstamp) if args.bitstamp else None

    df = _combine_sources(coinbase_df, bitstamp_df, args.source)
    df, feature_names = _select_features(df)

    values = df.to_numpy(dtype="float64")
    close_idx = feature_names.index("close")

    X, y = _build_sequences(
        values=values,
        close_index=close_idx,
        seq_len=int(args.seq_len),
        horizon=int(args.horizon),
    )

    X_train, y_train, X_val, y_val, X_test, y_test = _time_split(
        X, y, float(args.train_ratio), float(args.val_ratio)
    )

    mean, std = _fit_standardizer(X_train)

    X_train = _apply_standardizer(X_train, mean, std)
    X_val = _apply_standardizer(X_val, mean, std)
    X_test = _apply_standardizer(X_test, mean, std)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        args.out,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        mean=mean,
        std=std,
        feature_names=np.array(feature_names),
        seq_len=np.array([int(args.seq_len)], dtype=np.int32),
        horizon=np.array([int(args.horizon)], dtype=np.int32),
        source=np.array([args.source]),
    )


if __name__ == "__main__":
    main()
