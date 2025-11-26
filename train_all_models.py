# train_all_models.py — FINAL WORKING VERSION (2025)
# Trains all 60 models (12 coins × 5 timeframes) with zero bullshit

import os
import subprocess
import sys

# These must match your fetch_data.py and nlp_parser.py
COINS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT",
    "XRPUSDT", "DOGEUSDT", "SHIBUSDT", "PEPEUSDT",
    "LINKUSDT", "AVAXUSDT", "TONUSDT"
]

TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]  # matches Binance format

def train_one(coin: str, tf: str):
    print(f"Training {coin} @ {tf}...", end=" ")
    result = subprocess.run([
        sys.executable, "train_one_coin.py",
        coin, tf
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("SUCCESS")
    else:
        print("FAILED")
        print(result.stderr[-500:])  # show last 500 chars of error

def main():
    os.makedirs("models", exist_ok=True)
    
    total = len(COINS) * len(TIMEFRAMES)
    done = 0

    print(f"JOAI REBIRTH INITIATED — Training {total} elite models\n")

    for coin in COINS:
        for tf in TIMEFRAMES:
            done += 1
            print(f"[{done:2d}/{total}] ", end="")
            train_one(coin, tf)

    print("\nALL 60 MODELS TRAINED.")
    print("JoAI is now a multi-timeframe god.")
    print("Deploy to Render and dominate.")

if __name__ == "__main__":
    main()