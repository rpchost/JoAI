# train_one_coin.py — WITH .ENV LOADING

import sys
import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import psycopg2
import pandas as pd
from dotenv import load_dotenv

# Force UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables from .env file
load_dotenv()

SEQUENCE_LENGTH = 60
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi',
    'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower', 'atr',
    'stoch_k', 'stoch_d', 'willr', 'volume_sma', 'roc'
]

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """EXACT SAME AS PREDICTION — 100% MATCH"""
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

def get_data(symbol: str, db_tf: str):
    # Safe DATABASE_URL retrieval
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable not set! Create a .env file or set it manually.")
    
    conn = psycopg2.connect(db_url + "?sslmode=require")
    query = """
        SELECT open, high, low, close, volume
        FROM crypto_candles
        WHERE symbol = %s AND timeframe = %s
        ORDER BY timestamp ASC
        LIMIT 3000
    """
    df = pd.read_sql(query, conn, params=(symbol, db_tf))
    conn.close()

    if len(df) < 100:
        raise ValueError(f"Not enough data: only {len(df)} rows found")

    df = calculate_indicators(df)
    return df[FEATURE_COLUMNS].values.astype(float)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 3])  # close price
    return np.array(X), np.array(y)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_one_coin.py BTCUSDT 1h")
        sys.exit(1)

    symbol = sys.argv[1].upper()
    tf = sys.argv[2]

    TIMEFRAME_TO_DB = {
        "1m": "1minute", "5m": "5minutes", "15m": "15minutes",
        "1h": "1hour", "4h": "4hours"
    }
    db_tf = TIMEFRAME_TO_DB[tf]

    print(f"=== TRAINING {symbol} @ {tf} ===")
    print(f"Current working directory: {os.getcwd()}")
    
    try:
        data = get_data(symbol, db_tf)
        print(f"✓ Loaded {len(data)} rows with indicators")

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)
       
        X, y = create_sequences(scaled, SEQUENCE_LENGTH)
        y = y.copy()  # y is now raw close prices (no scaling!)
        target_scaler = None  # we will save None or a dummy

        print(f"✓ Created {len(X)} training sequences")

        split = int(0.8 * len(X))
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(FEATURE_COLUMNS))),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile('adam', 'mse')
        print(f"✓ Model compiled, training on {split} samples...")
        
        model.fit(X[:split], y[:split], validation_data=(X[split:], y[split:]),
                  epochs=200, batch_size=32,
                  callbacks=[EarlyStopping(patience=10, restore_best_weights=True)], 
                  verbose=0)
        print(f"✓ Training completed")

        # Absolute path resolution
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, f"saved_model_{symbol}_{db_tf}.keras")
        scaler_path = model_path.replace(".keras", "_scaler.pkl")
        target_scaler_path = model_path.replace(".keras", "_target_scaler.pkl")
        
        print(f"Saving to: {model_path}")
        model.save(model_path)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"FAILED TO SAVE: {model_path}")
        
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        
        with open(target_scaler_path, "wb") as f:
            pickle.dump(target_scaler, f)
        
        # Verify all files
        model_size = os.path.getsize(model_path) / 1024  # KB
        scaler_size = os.path.getsize(scaler_path) / 1024
        target_size = os.path.getsize(target_scaler_path) / 1024
        
        print(f"✓ Model saved: {model_size:.1f} KB")
        print(f"✓ Scaler saved: {scaler_size:.1f} KB")
        print(f"✓ Target scaler saved: {target_size:.1f} KB")
        print(f"SUCCESS → {model_path}")

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)