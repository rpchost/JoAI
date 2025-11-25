# train_one_coin.py — WORKS WITH YOUR CURRENT lstm_model.py (NO CLASS NEEDED)
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from models.lstm_model import predict_next_candle  # just to force import
from models.lstm_model import _get_model_paths, load_model_and_scalers
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from utils.indicators import add_technical_indicators

# CONFIG
SYMBOL = "XRPUSDT"  # Change this to train other coins
TIMEFRAMES = ["1 minute", "5 minutes", "15 minutes", "1 hour", "4 hours"]
SEQUENCE_LENGTH = 60
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi',
    'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower', 'atr',
    'stoch_k', 'stoch_d', 'willr', 'volume_sma', 'roc'
]

def fetch_data(symbol: str):
    # Copy-paste from your current get_latest_data() but with more rows
    import psycopg2
    from dotenv import load_dotenv
    load_dotenv()
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    query = "SELECT timestamp, open, high, low, close, volume FROM crypto_candles WHERE symbol = %s ORDER BY timestamp ASC"
    df = pd.read_sql(query, conn, params=(symbol,))
    conn.close()
    return df

def train_model_for_timeframe(symbol: str, tf_name: str):
    tf_key = tf_name.replace(" ", "")
    model_path = f"models/saved_model_{symbol}_{tf_key}.keras"
    scaler_path = f"models/saved_model_{symbol}_{tf_key}_scaler.pkl"
    target_scaler_path = f"models/saved_model_{symbol}_{tf_key}_target_scaler.pkl"

    if os.path.exists(model_path):
        print(f"Already exists → {model_path}")
        return

    print(f"Training {symbol} @ {tf_name}...")

    df = fetch_data(symbol)
    df = add_technical_indicators(df)
    df = df.dropna()

    data = df[FEATURE_COLUMNS].values
    target = df['close'].values[SEQUENCE_LENGTH:]

    # Scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    target_scaler = MinMaxScaler()
    scaled_target = target_scaler.fit_transform(target.reshape(-1, 1))

    # Create sequences
    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_data)):
        X.append(scaled_data[i-SEQUENCE_LENGTH:i])
        y.append(scaled_target[i-SEQUENCE_LENGTH])

    X = np.array(X)
    y = np.array(y)

    # Model
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(SEQUENCE_LENGTH, len(FEATURE_COLUMNS))))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    # Train
    model.fit(X, y, epochs=80, batch_size=32, validation_split=0.2,
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)], verbose=1)

    # Save
    model.save(model_path)
    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(target_scaler_path, 'wb') as f:
        pickle.dump(target_scaler, f)

    print(f"SAVED: {model_path}")

# MAIN
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    print(f"TRAINING {SYMBOL} ACROSS 5 TIMEFRAMES...")
    for tf in TIMEFRAMES:
        train_model_for_timeframe(SYMBOL, tf)
    print(f"{SYMBOL} IS NOW ELITE")