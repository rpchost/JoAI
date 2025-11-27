# train_all_models.py
# Run this once after adding new coins or updating data
# Trains a separate high-quality LSTM model for EVERY supported coin

import os
from models.lstm_model import LSTMCryptoPredictor

# MUST match your symbol_map in nlp_parser.py and fetch_data.py
SUPPORTED_COINS = [
    "BTCUSD",
    "ETHUSD",
    "BNBUSD",
    "ADAUSD",
    "SOLUSD",
    "XRPUSD",
    "DOGEUSD",
    "SHIBUSD",
    "PEPEUSD",
    "LINKUSD",
    "AVAXUSD",
    "TONUSD"
]

def train_all_models():
    print(f"Starting training for {len(SUPPORTED_COINS)} cryptocurrencies...\n")
    print("=" * 70)

    successful = []
    failed = []

    for i, symbol in enumerate(SUPPORTED_COINS, 1):
        print(f"[{i:2d}/{len(SUPPORTED_COINS)}] Training model for {symbol}...", end=" ")

        # Each coin gets its own model + scaler files
        model_path = f"models/saved_model_{symbol}.keras"

        try:
            predictor = LSTMCryptoPredictor(
                symbol=symbol,
                model_path=model_path,
                sequence_length=60
            )

            # This will fetch data, add indicators, train, save model + scalers
            predictor.train(epochs=120, batch_size=32, validation_split=0.2)
            
            print("DONE")
            successful.append(symbol)
        except Exception as e:
            print(f"FAILED → {str(e)}")
            failed.append(f"{symbol}: {str(e)}")

        print("-" * 70)

    # Final summary
    print("\nTRAINING COMPLETE!")
    print(f"Successful: {len(successful)}/{len(SUPPORTED_COINS)}")
    if successful:
        print("Trained coins:", ", ".join(successful))
    if failed:
        print("Failed coins:")
        for f in failed:
            print(f"  • {f}")
    else:
        print("All models trained perfectly! JoAI is now a multi-coin prediction monster!")

if __name__ == "__main__":
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Run training
    train_all_models()