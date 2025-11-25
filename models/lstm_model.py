# lstm_model.py — FINAL BULLETPROOF VERSION (2025)
import os
import pickle
import psycopg2
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

# Load env
load_dotenv()

# === CONFIG ===
MODEL_DIR = "models"
SEQUENCE_LENGTH = 60
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi',
    'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower', 'atr',
    'stoch_k', 'stoch_d', 'willr', 'volume_sma', 'roc'
]

# Ensure models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# === PER-SYMBOL MODEL CACHE ===
_model_cache = {}  # {symbol: {"model": ..., "scaler": ..., "target_scaler": ...}}


def _get_model_paths(symbol: str, timeframe: str = "1 hour"):
    """NEW: supports both old and new naming, but prefers timeframe-specific"""
    tf_key = timeframe.lower()
    tf_key = tf_key.replace("minutes", "minute").replace("hours", "hour").replace(" ", "")
    
    # First try: timeframe-specific model (NEW)
    base = os.path.join(MODEL_DIR, f"saved_model_{symbol}_{tf_key}")
    if os.path.exists(f"{base}.keras"):
        return {
            "model": f"{base}.keras",
            "scaler": f"{base}_scaler.pkl",
            "target_scaler": f"{base}_target_scaler.pkl"
        }
    
    # Fallback: old generic model (only if you forgot to delete)
    old_base = os.path.join(MODEL_DIR, f"saved_model_{symbol}")
    return {
        "model": f"{old_base}.keras",
        "scaler": f"{old_base}_scaler.pkl",
        "target_scaler": f"{old_base}_target_scaler.pkl"
    }


def load_model_and_scalers(symbol: str, timeframe: str = "1h"):
    key = f"{symbol}_{timeframe}"
    if key in _model_cache:
        return _model_cache[key]

    paths = _get_model_paths(symbol, timeframe)

    if not os.path.exists(paths["model"]):
        raise FileNotFoundError(f"Model not found: {paths['model']}\nRun: python train_all_models.py --timeframes")

    from tensorflow.keras.models import load_model
    model = load_model(paths["model"])
    with open(paths["scaler"], "rb") as f:
        scaler = pickle.load(f)
    with open(paths["target_scaler"], "rb") as f:
        target_scaler = pickle.load(f)

    _model_cache[key] = {"model": model, "scaler": scaler, "target_scaler": target_scaler}
    return _model_cache[key]


def get_latest_data(symbol: str, limit: int = SEQUENCE_LENGTH + 20):
    """Fetch latest candles from DB"""
    conn_str = os.getenv("DATABASE_URL")
    if not conn_str:
        raise ValueError("DATABASE_URL not set!")

    conn = psycopg2.connect(conn_str, connect_timeout=10)
    try:
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM crypto_candles
            WHERE symbol = %s
            ORDER BY timestamp DESC
            LIMIT %s
        """
        df = pd.read_sql(query, conn, params=(symbol, limit))
    finally:
        conn.close()

    if len(df) < SEQUENCE_LENGTH:
        raise ValueError(f"Not enough data for {symbol}: {len(df)} rows")

    df = df.iloc[::-1].reset_index(drop=True)  # oldest first
    return df


def predict_next_candle(symbol: str, timeframe: str = "1 hour"):
    symbol = symbol.upper()
    tf_normalized = timeframe.lower().replace("minutes", "minute").replace("hours", "hour")

    tf_key = timeframe.lower()
    tf_key = tf_key.replace("minutes", "minute").replace("hours", "hour")
    tf_key = tf_key.replace(" ", "")  # ← REMOVE ALL SPACES!
    tf_map = {
        "1minute": "1minute",
        "5minute": "5minutes",     # ← YOUR TRAINED FILES USE "5minutes"
        "15minute": "15minutes",
        "1hour": "1hour",
        "4hour": "4hours",
        "1day": "1day"
    }
    final_key = tf_map.get(tf_key, "1hour")  # default to 1h
    assets = load_model_and_scalers(symbol, final_key)    
    #assets = load_model_and_scalers(symbol, tf_normalized)
    model = assets["model"]
    scaler = assets["scaler"]
    target_scaler = assets["target_scaler"]

    df = get_latest_data(symbol)
    from utils.indicators import add_technical_indicators
    df = add_technical_indicators(df)

    latest = df.tail(SEQUENCE_LENGTH)[FEATURE_COLUMNS].values
    scaled = scaler.transform(latest)
    X = scaled.reshape((1, SEQUENCE_LENGTH, len(FEATURE_COLUMNS)))

    pred_scaled = model.predict(X, verbose=0)[0][0]
    pred_close = float(target_scaler.inverse_transform([[pred_scaled]])[0][0])
    last_close = float(df["close"].iloc[-1])

    change = (pred_close - last_close) / last_close
    volatility = df["close"].pct_change().std() * 3

    pred_open = last_close
    pred_high = max(pred_close * (1 + abs(change) * 0.6 + volatility), pred_close, pred_open)
    pred_low = min(pred_close * (1 - abs(change) * 0.6 - volatility), pred_close, pred_open)
    pred_volume = float(df["volume"].tail(20).mean())

    return {
        "open": round(pred_open, 2),
        "high": round(pred_high, 2),
        "low": round(pred_low, 2),
        "close": round(pred_close, 2),
        "volume": round(pred_volume, 0)
    }

# Optional: Clear cache (for hot reloads)
def clear_model_cache():
    global _model_cache
    _model_cache.clear()
    import gc; gc.collect()