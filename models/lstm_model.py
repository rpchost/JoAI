# lstm_model.py — FIXED PREDICTION VERSION (2025)
import os
import pickle
import psycopg2
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

load_dotenv()

# CONFIG
MODEL_DIR = "models"
SEQUENCE_LENGTH = 60
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi',
    'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower', 'atr',
    'stoch_k', 'stoch_d', 'willr', 'volume_sma', 'roc'
]

os.makedirs(MODEL_DIR, exist_ok=True)

def load_model_and_scalers(symbol: str, db_tf: str):
    model_path = f"{MODEL_DIR}/saved_model_{symbol}_{db_tf}.keras"
    scaler_path = model_path.replace(".keras", "_scaler.pkl")
    target_scaler_path = model_path.replace(".keras", "_target_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model missing: {model_path}")

    model = load_model(model_path)
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    with open(target_scaler_path, "rb") as f:
        target_scaler = pickle.load(f)
    
    # Verify target_scaler is valid
    if target_scaler is None or not hasattr(target_scaler, 'inverse_transform'):
        raise ValueError(f"Invalid target_scaler for {symbol}_{db_tf}. Retrain the model!")
    
    return model, scaler, target_scaler

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """EXACT SAME as training — 100% match"""
    c = df['close']
    h = df['high']
    l = df['low']
    v = df['volume']

    df = df.copy()
    df['sma_20'] = c.rolling(20).mean()
    df['sma_50'] = c.rolling(50).mean()
    df['ema_12'] = c.ewm(span=12, adjust=False).mean()
    df['ema_26'] = c.ewm(span=26, adjust=False).mean()

    delta = c.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(com=13, adjust=False).mean()
    roll_down = down.ewm(com=13, adjust=False).mean()
    rs = roll_up / roll_down
    df['rsi'] = 100 - (100 / (1 + rs))

    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    std20 = c.rolling(20).std()
    df['bb_middle'] = df['sma_20']
    df['bb_upper'] = df['sma_20'] + 2 * std20
    df['bb_lower'] = df['sma_20'] - 2 * std20

    tr = pd.DataFrame(index=df.index)
    tr['h_l'] = h - l
    tr['h_pc'] = abs(h - c.shift())
    tr['l_pc'] = abs(l - c.shift())
    df['atr'] = tr.max(axis=1).rolling(14).mean()

    df['stoch_k'] = 100 * (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min() + 1e-8)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['willr'] = -100 * (h.rolling(14).max() - c) / (h.rolling(14).max() - l.rolling(14).min() + 1e-8)
    df['volume_sma'] = v.rolling(20).mean()
    df['roc'] = c.pct_change(10) * 100

    df = df.fillna(0)
    return df

def predict_next_candle(symbol: str, timeframe: str):
    symbol = symbol.upper()
    
    # Map user timeframe → DB timeframe
    tf_map = {
        "1 minute": "1minute", "5 minutes": "5minutes", "15 minutes": "15minutes",
        "1 hour": "1hour", "4 hours": "4hours", "1 day": "1day"
    }
    db_tf = tf_map.get(timeframe, "1hour")

    # Fetch raw data
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return "Error: DATABASE_URL not configured"
    
    #conn = psycopg2.connect(db_url + "?sslmode=require")
    query = """
        SELECT open, high, low, close, volume
        FROM crypto_candles
        WHERE symbol = %s AND timeframe = %s
        ORDER BY timestamp DESC LIMIT 300
    """
    #df = pd.read_sql(query, conn, params=(symbol, db_tf))
    from sqlalchemy import create_engine
    engine = create_engine(db_url + "?sslmode=require")
    df = pd.read_sql(query, engine, params=(symbol, db_tf))

    #conn.close()

    if len(df) < 100:
        return "Not enough data"

    df = df.iloc[::-1].reset_index(drop=True)  # oldest first
    df = calculate_indicators(df)

    data = df[FEATURE_COLUMNS].tail(60).values.astype(float)

    try:
        model, scaler, target_scaler = load_model_and_scalers(symbol, db_tf)
        
        # Scale input features
        scaled = scaler.transform(data)
        seq = scaled.reshape(1, 60, -1)
        
        # Get prediction (this is a SCALED value between 0-1)
        pred_scaled = model.predict(seq, verbose=0)
        
        # CRITICAL: Inverse transform to get REAL price
        pred_price = target_scaler.inverse_transform(pred_scaled)[0][0]
        
        # Sanity check
        if pred_price < 0 or pred_price > 1000000:
            return f"Error: Unrealistic prediction ({pred_price:.2f}). Model needs retraining."
        
        return f"${pred_price:,.2f}"
        
    except Exception as e:
        return f"Error: {str(e)}"