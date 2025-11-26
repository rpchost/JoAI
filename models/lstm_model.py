# lstm_model.py — FINAL BULLETPROOF VERSION (2025)
import os
import pickle
import psycopg2
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

from tensorflow.keras.models import load_model

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


# def _get_model_paths(symbol: str, timeframe: str = "1 hour"):
#     """NEW: supports both old and new naming, but prefers timeframe-specific"""
#     tf_key = timeframe.lower()
#     tf_key = tf_key.replace("minutes", "minute").replace("hours", "hour").replace(" ", "")
    
#     # First try: timeframe-specific model (NEW)
#     base = os.path.join(MODEL_DIR, f"saved_model_{symbol}_{tf_key}")
#     if os.path.exists(f"{base}.keras"):
#         return {
#             "model": f"{base}.keras",
#             "scaler": f"{base}_scaler.pkl",
#             "target_scaler": f"{base}_target_scaler.pkl"
#         }
    
#     # Fallback: old generic model (only if you forgot to delete)
#     old_base = os.path.join(MODEL_DIR, f"saved_model_{symbol}")
#     return {
#         "model": f"{old_base}.keras",
#         "scaler": f"{old_base}_scaler.pkl",
#         "target_scaler": f"{old_base}_target_scaler.pkl"
#     }


def load_model_and_scalers(symbol: str, timeframe_key: str):
    """timeframe_key must be the exact suffix like '5minutes', '1hour' etc."""
    key = f"{symbol}_{timeframe_key}"
    if key in _model_cache:
        return _model_cache[key]

    base = os.path.join(MODEL_DIR, f"saved_model_{symbol}_{timeframe_key}")
    paths = {
        "model": f"{base}.keras",
        "scaler": f"{base}_scaler.pkl",
        "target_scaler": f"{base}_target_scaler.pkl"
    }

    if not os.path.exists(paths["model"]):
        raise FileNotFoundError(f"MODEL NOT FOUND: {paths['model']}")

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

def predict_next_candle(symbol: str, timeframe: str, latest_data: np.ndarray):
    symbol_clean = symbol.replace("/", "").upper()
    
    # MAP USER INPUT → DB FORMAT (same as training)
    TIMEFRAME_TO_DB = {
        "1 minute":   "1minute",
        "5 minutes":  "5minutes",
        "15 minutes": "15minutes",
        "1 hour":     "1hour",
        "4 hours":    "4hours",
        "1 day":      "1day"
    }
    db_tf = TIMEFRAME_TO_DB.get(timeframe, "1hour")

    model_path = f"models/saved_model_{symbol_clean}_{db_tf}.keras"

    if not os.path.exists(model_path):
        return "Model not trained yet — run: python train_all_models.py"

    try:
        model = load_model(model_path)
        scaler_path = model_path.replace(".keras", "_scaler.pkl")
        target_scaler_path = model_path.replace(".keras", "_target_scaler.pkl")

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(target_scaler_path, "rb") as f:
            target_scaler = pickle.load(f)

        scaled = scaler.transform(latest_data)
        seq = scaled[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, -1)
        pred_scaled = model.predict(seq, verbose=0)
        pred_price = target_scaler.inverse_transform(pred_scaled)[0][0]

        return f"Predicted next close: ${pred_price:,.6f}"

    except Exception as e:
        return f"Prediction error: {str(e)}"
    
# Optional: Clear cache (for hot reloads)
def clear_model_cache():
    global _model_cache
    _model_cache.clear()
    import gc; gc.collect()