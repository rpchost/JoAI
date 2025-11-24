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


def _get_model_paths(symbol: str):
    """Return exact paths for a symbol — NO FALLBACKS EVER"""
    base = os.path.join(MODEL_DIR, f"saved_model_{symbol}")
    return {
        "model": f"{base}.keras",
        "scaler": f"{base}_scaler.pkl",
        "target_scaler": f"{base}_target_scaler.pkl"
    }


def load_model_and_scalers(symbol: str):
    """Load model + scalers for a symbol — raises clear error if missing"""
    if symbol in _model_cache:
        return _model_cache[symbol]

    paths = _get_model_paths(symbol)

    if not os.path.exists(paths["model"]):
        raise FileNotFoundError(
            f"Model NOT found for {symbol}!\n"
            f"Expected: {paths['model']}\n"
            f"Run: python train_all_models.py first!"
        )

    if not os.path.exists(paths["scaler"]) or not os.path.exists(paths["target_scaler"]):
        raise FileNotFoundError(
            f"Scalers missing for {symbol}!\n"
            f"Missing: {paths['scaler']} or {paths['target_scaler']}"
        )

    # Lazy import TF only when needed
    from tensorflow.keras.models import load_model

    print(f"Loading model for {symbol}...")
    model = load_model(paths["model"])
    with open(paths["scaler"], "rb") as f:
        scaler = pickle.load(f)
    with open(paths["target_scaler"], "rb") as f:
        target_scaler = pickle.load(f)

    # Cache it
    _model_cache[symbol] = {
        "model": model,
        "scaler": scaler,
        "target_scaler": target_scaler
    }

    return _model_cache[symbol]


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


def predict_next_candle(symbol: str = "BTCUSDT"):
    """
    MAIN PREDICTION FUNCTION — CLEAN, SAFE, NO FALLBACKS
    """
    symbol = symbol.upper()

    # === 1. Load model + scalers ===
    try:
        assets = load_model_and_scalers(symbol)
    except FileNotFoundError as e:
        raise Exception(f"Model not trained yet for {symbol}. Please train the model first.") from e

    model = assets["model"]
    scaler = assets["scaler"]
    target_scaler = assets["target_scaler"]

    # === 2. Get latest data ===
    df = get_latest_data(symbol)

    # === 3. Add indicators ===
    from utils.indicators import add_technical_indicators
    df = add_technical_indicators(df)

    # === 4. Prepare input sequence ===
    latest = df.tail(SEQUENCE_LENGTH)[FEATURE_COLUMNS].values
    scaled = scaler.transform(latest)
    X = scaled.reshape((1, SEQUENCE_LENGTH, len(FEATURE_COLUMNS)))

    # === 5. Predict ===
    pred_scaled = model.predict(X, verbose=0)[0][0]
    pred_close = float(target_scaler.inverse_transform([[pred_scaled]])[0][0])
    last_close = float(df["close"].iloc[-1])

    # === 6. Generate realistic OHLC ===
    change = (pred_close - last_close) / last_close
    volatility = df["close"].pct_change().std() * 3  # rough volatility

    pred_open = last_close
    pred_high = pred_close * (1 + abs(change) * 0.6 + volatility)
    pred_low = pred_close * (1 - abs(change) * 0.6 - volatility)
    pred_volume = float(df["volume"].tail(20).mean())

    # Enforce OHLC logic
    pred_high = max(pred_high, pred_close, pred_open)
    pred_low = min(pred_low, pred_close, pred_open)

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