# train_one_coin.py — FINAL WINDOWS-PROOF VERSION (NO UNICODE EVER)

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

SEQUENCE_LENGTH = 60
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi',
    'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower', 'atr',
    'stoch_k', 'stoch_d', 'willr', 'volume_sma', 'roc'
]

def get_data(symbol: str, timeframe: str):
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL not found -> Running in LOCAL TEST MODE (fake data)")
        fake = np.random.rand(3000, len(FEATURE_COLUMNS))
        fake[:, 3] = np.cumsum(np.random.randn(3000) * 0.001 + 1) * 60000
        return fake

    if "sslmode" not in db_url:
        db_url += "?sslmode=require"

    conn = psycopg2.connect(db_url)
    query = f"SELECT {', '.join(FEATURE_COLUMNS)} FROM crypto_candles_with_indicators WHERE symbol=%s AND timeframe=%s ORDER BY timestamp LIMIT 3000"
    df = pd.read_sql(query, conn, params=(symbol, timeframe))
    conn.close()
    return df[FEATURE_COLUMNS].values.astype(float)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 3])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_one_coin.py BTCUSDT 1h")
        sys.exit(1)

    symbol = sys.argv[1].upper()
    tf = sys.argv[2]
    tf_map = {"1m": "1minute", "5m": "5minutes", "15m": "15minutes", "1h": "1hour", "4h": "4hours"}
    db_tf = tf_map.get(tf, "1hour")

    print(f"Training {symbol} @ {tf} -> DB: {db_tf}")

    try:
        data = get_data(symbol, db_tf)
        print(f"Loaded {len(data)} rows")

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)
        target_scaler = MinMaxScaler().fit(data[:, 3].reshape(-1, 1))

        X, y = create_sequences(scaled, SEQUENCE_LENGTH)
        y = target_scaler.transform(y.reshape(-1, 1)).flatten()

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
        model.fit(X[:split], y[:split], validation_data=(X[split:], y[split:]), epochs=200, batch_size=32,
                  callbacks=[EarlyStopping(patience=10, restore_best_weights=True)], verbose=1)

        os.makedirs("models", exist_ok=True)
        model.save(f"models/saved_model_{symbol}_{tf}.keras")
        pickle.dump(scaler, open(f"models/saved_model_{symbol}_{tf}_scaler.pkl", "wb"))
        pickle.dump(target_scaler, open(f"models/saved_model_{symbol}_{tf}_target_scaler.pkl", "wb"))
        print(f"SAVED → models/saved_model_{symbol}_{tf}.keras")

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()