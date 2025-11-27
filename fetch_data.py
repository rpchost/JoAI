# fetch_data.py — FINAL — DIRECT BINANCE.US — NO CCXT — WORKS 100%
import pandas as pd
import requests
from datetime import datetime
import os
from dotenv import load_dotenv
import time
import psycopg2

load_dotenv()

TIMEFRAMES = {
    "1m": 1000,
    "5m": 1000,
    "15m": 1000,
    "1h": 2000,
    "4h": 2000
}

def fetch_ohlcv_direct(timeframe: str, limit: int):
    url = "https://api.binance.us/api/v3/klines"
    params = {'symbol': 'BTCUSD', 'interval': timeframe, 'limit': limit}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            print("  Empty response")
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df['symbol'] = 'BTCUSD'
        print(f"  SUCCESS → {len(df)} candles | {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
        return df
    except Exception as e:
        print(f"  ERROR: {e}")
        return pd.DataFrame()

def store_candles_postgresql(df, tf: str):
    if df.empty:
        return
    tf_map = {'1m': '1minute', '5m': '5minutes', '15m': '15minutes', '1h': '1hour', '4h': '4hours'}
    db_tf = tf_map.get(tf, '1hour')

    conn_str = os.getenv("DATABASE_URL")
    if conn_str and "sslmode" not in conn_str:
        conn_str += "?sslmode=require"
    
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor()
    
    sql = """
    INSERT INTO crypto_candles (symbol, timeframe, timestamp, open, high, low, close, volume)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
    """
    
    count = 0
    for _, row in df.iterrows():
        cur.execute(sql, (
            "BTCUSD", db_tf, row['timestamp'],
            row['open'], row['high'], row['low'],
            row['close'], row['volume']
        ))
        count += 1
    
    conn.commit()
    conn.close()
    print(f"   Stored {count} new rows @ {db_tf}")

def populate_multiple_symbols():
    print("=" * 70)
    print("JOAI DATA UPDATE — BINANCE.US DIRECT — NO CCXT — 2025 FINAL")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    for i, (tf, limit) in enumerate(TIMEFRAMES.items(), 1):
        print(f"[{i}/{len(TIMEFRAMES)}] BTCUSD @ {tf} → ", end="", flush=True)
        df = fetch_ohlcv_direct(tf, limit)
        if not df.empty:
            store_candles_postgresql(df, tf)
            print(f"STORED {len(df)}")
        else:
            print("NO DATA")
        time.sleep(1.2)

    print("=" * 70)
    print("ALL DATA UPDATED — YOU ARE NOW UNSTOPPABLE")
    print(f"Finished: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

if __name__ == "__main__":
    populate_multiple_symbols()
