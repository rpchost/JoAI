# train_all_models.py — FULL 60 MODELS VERSION (2025)
# Trains all 60 models (12 coins × 5 timeframes)

import os
import subprocess
import sys
from datetime import datetime

# All 12 coins (uncomment to train full suite)
COINS = [
    "BTCUSD",
    # "ETHUSD",
    # "BNBUSD",
    # "ADAUSD",
    # "SOLUSD",
    # "XRPUSD",
    # "DOGEUSD",
    # "SHIBUSD",
    # "PEPEUSD",
    # "LINKUSD",
    # "AVAXUSD",
    # "TONUSD"
]

# All timeframes
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]

def train_one(coin: str, tf: str):
    """Train a single model and return success status"""
    print(f"Training {coin:12s} @ {tf:4s}... ", end="", flush=True)
    start = datetime.now()
    
    result = subprocess.run([
        sys.executable, "train_one_coin.py",
        coin, tf
    ], capture_output=True, text=True)

    duration = (datetime.now() - start).total_seconds()
    
    if result.returncode == 0:
        print(f"✓ SUCCESS ({duration:.1f}s)")
        return True
    else:
        print(f"✗ FAILED ({duration:.1f}s)")
        # Show last 500 chars of error
        if result.stderr:
            print(f"  Error: {result.stderr[-500:]}")
        return False

def main():
    os.makedirs("models", exist_ok=True)
    
    total = len(COINS) * len(TIMEFRAMES)
    success_count = 0
    failed = []
    
    print("=" * 70)
    print(f"JOAI REBIRTH INITIATED — Training {total} elite models")
    print(f"Coins: {len(COINS)} | Timeframes: {len(TIMEFRAMES)}")
    print("=" * 70)
    print()
    
    overall_start = datetime.now()
    
    for i, coin in enumerate(COINS, 1):
        print(f"\n[COIN {i}/{len(COINS)}] {coin}")
        print("-" * 70)
        
        for j, tf in enumerate(TIMEFRAMES, 1):
            model_num = (i - 1) * len(TIMEFRAMES) + j
            print(f"[{model_num:2d}/{total}] ", end="")
            
            if train_one(coin, tf):
                success_count += 1
            else:
                failed.append(f"{coin}@{tf}")
    
    # Summary
    overall_duration = (datetime.now() - overall_start).total_seconds()
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"✓ Successful: {success_count}/{total}")
    print(f"✗ Failed:     {len(failed)}/{total}")
    print(f"⏱ Total time: {overall_duration/60:.1f} minutes")
    
    if failed:
        print("\nFailed models:")
        for model in failed:
            print(f"  - {model}")
        print("\nRe-run those individually with:")
        print("  python train_one_coin.py <COIN> <TIMEFRAME>")
    else:
        print("\n🎉 ALL 60 MODELS TRAINED SUCCESSFULLY!")
        print("JoAI is now a multi-timeframe god.")
        print("Deploy to Render and dominate.")
    
    print("=" * 70)

if __name__ == "__main__":
    main()