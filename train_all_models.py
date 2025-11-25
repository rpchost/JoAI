# train_all_models.py
# Run this once after adding new coins or updating data
# Trains a separate high-quality LSTM model for EVERY supported coin

import os
from models.lstm_model import LSTMCryptoPredictor

# MUST match your symbol_map in nlp_parser.py and fetch_data.py
SUPPORTED_COINS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "ADAUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "SHIBUSDT",
    "PEPEUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "TONUSDT"
]

TIMEFRAMES = ["1 minute", "5 minutes", "15 minutes", "1 hour", "4 hours"]

class LSTMCryptoPredictor:
    def __init__(self, symbol, model_path, sequence_length=60, timeframe="1h"):
        self.symbol = symbol
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.timeframe = timeframe.replace(" ", "")

def train_all_models():
    total = len(SUPPORTED_COINS) * len(TIMEFRAMES)
    count = 0

    for symbol in SUPPORTED_COINS:
        for tf in TIMEFRAMES:
            count += 1
            tf_key = tf.replace(" ", "")
            model_path = f"models/saved_model_{symbol}_{tf_key}.keras"

            print(f"[{count:3d}/{total}] Training {symbol} @ {tf} → {model_path}")

            try:
                predictor = LSTMCryptoPredictor(
                    symbol=symbol,
                    model_path=model_path,
                    timeframe=tf
                )
                predictor.train(epochs=100, batch_size=32)  # slightly less epochs per model
                print("DONE")
            except Exception as e:
                print(f"FAILED: {e}")

    print("JOAI IS NOW A TIMEFRAME GOD.")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_all_models()