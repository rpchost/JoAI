# fetch_data.py — FINAL MULTI-TIMEFRAME DATA HARVESTER (2025)

import ccxt
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import time
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === CONFIG ===
TIMEFRAMES = {
    "1m": {"limit": 1000, "desc": "1 minute"},
    "5m": {"limit": 1000, "desc": "5 minutes"},
    "15m": {"limit": 1000, "desc": "15 minutes"},
    "1h": {"limit": 2000, "desc": "1 hour"},
    "4h": {"limit": 2000, "desc": "4 hours"}
}

# SYMBOLS = [
#     "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT",
#     "XRP/USDT", "DOGE/USDT", "SHIB/USDT", "PEPE/USDT",
#     "LINK/USDT", "AVAX/USDT", "TON/USDT"
# ]

SYMBOLS = [
   "BTC/USDT"
]

def get_latest_timestamp_in_db(symbol: str) -> int:
    """Return the latest timestamp (in ms) already stored for this symbol, or 0 if none"""
    import psycopg2
    from dotenv import load_dotenv
    import os
    load_dotenv()

    conn_str = os.getenv("DATABASE_URL")
    if "sslmode" not in conn_str:
        conn_str += "?sslmode=require"

    conn = psycopg2.connect(conn_str)
    cur = conn.cursor()
    cur.execute(
        "SELECT COALESCE(MAX(timestamp), '1970-01-01'::timestamp) FROM crypto_candles WHERE symbol = %s",
        (symbol,)
    )
    latest_dt = cur.fetchone()[0]
    conn.close()

    # Convert to milliseconds (Binance format)
    if latest_dt == datetime(1970, 1, 1):
        return 0
    return int(latest_dt.timestamp() * 1000)


def fetch_candles(symbol: str, timeframe: str, limit: int = 1000):
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
        'timeout': 30000,
    })

    symbol_clean = symbol.replace("/", "")
    print(f"Fetching latest {limit} {timeframe} candles for {symbol}...")

    try:
        # NO 'since' → Binance always returns the most recent candles first
        candles = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    except Exception:
        print("  Binance rate-limited or error → retrying once...")
        time.sleep(2)
        candles = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    if not candles or len(candles) == 0:
        print("  No data received from Binance")
        return pd.DataFrame()

    # Binance returns newest first → reverse to chronological order
    candles.reverse()

    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["symbol"] = symbol_clean

    # Remove exact duplicates (just in case)
    before = len(df)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    print(f"  Received {before} → kept {len(df)} unique candles")
    print(f"  Date range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

    return df

def store_candles_postgresql(df, current_timeframe: str):
    """
    Store candles with timeframe column
    current_timeframe: '1m', '5m', '15m', '1h', '4h'
    """
    import psycopg2
    
    # Map ccxt timeframe → DB format
    tf_map = {
        '1m': '1minute',
        '5m': '5minutes',
        '15m': '15minutes',
        '1h': '1hour',
        '4h': '4hours',
        '1d': '1day'
    }
    db_timeframe = tf_map.get(current_timeframe, '1hour')

    conn_str = os.getenv("DATABASE_URL")
    if "sslmode" not in conn_str:
        conn_str += "?sslmode=require"
    
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor()
    
    # NEW SQL — NOW INCLUDES timeframe
    sql = """
    INSERT INTO crypto_candles (symbol, timeframe, timestamp, open, high, low, close, volume)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
    """
    
    count = 0
    for _, row in df.iterrows():
        cur.execute(sql, (
            row['symbol'],
            db_timeframe,                    # ← NEW: timeframe
            row['timestamp'],
            float(row['open']), float(row['high']), float(row['low']),
            float(row['close']), float(row['volume'])
        ))
        count += 1
    
    conn.commit()
    conn.close()
    print(f"   Stored {count} rows [{row['symbol']} @ {db_timeframe}]")

# def populate_all_data():
#     total = len(SYMBOLS) * len(TIMEFRAMES)
#     done = 0
    
#     print("JOAI DATA HARVESTER ACTIVATED — Feeding all 60 models (12 coins × 5 timeframes)")
#     print(f"Total candles to fetch: ~100,000+\n")
    
#     for symbol in SYMBOLS:
#         for tf, config in TIMEFRAMES.items():
#             desc = config["desc"]
#             limit = config["limit"]
            
#             print(f"[{done+1}/{total}] Fetching {symbol} @ {desc} ({limit} candles)...")
            
#             try:
#                 df = fetch_candles(symbol, tf, limit)
#                 if not df.empty:
#                     store_candles_postgresql(df, tf)  # ← NOW WITH timeframe
#                     print(f"   {symbol} {desc} → FED & STORED\n")
#                 else:
#                     print(f"   No new data for {symbol} {desc}\n")
#             except Exception as e:
#                 print(f"   FAILED: {e}\n")
            
#             done += 1
#             time.sleep(0.5)
    
#     print("ALL MODELS ARE NOW FULLY FED.")
#     print("You may now retrain: python train_one_coin.py")
# —————————————————————————————————————————————————————
# FINAL ENTRY POINT — USED BY GITHUB ACTIONS & LOCAL
# —————————————————————————————————————————————————————

def populate_multiple_symbols():
    """This function is called by update_data_local.py and GitHub Actions"""
    from datetime import datetime
    
    print("=" * 70)
    print("JOAI DATA UPDATE — GITHUB ACTIONS / LOCAL SCRIPT")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Updating {len(SYMBOLS)} coins × {len(TIMEFRAMES)} timeframes")
    print("=" * 70)

    total = len(SYMBOLS) * len(TIMEFRAMES)
    completed = 0

    for symbol in SYMBOLS:
        for tf, config in TIMEFRAMES.items():
            limit = config["limit"]
            completed += 1
            print(f"[{completed}/{total}] {symbol} @ {tf} → ", end="", flush=True)
            
            df = fetch_candles(symbol, tf, limit)
            if not df.empty:
                store_candles_postgresql(df, tf)
                print(f"STORED {len(df)} rows")
            else:
                print("No new data")
            
            time.sleep(0.4)  # Be gentle

    print("=" * 70)
    print("ALL DATA UPDATED SUCCESSFULLY")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


# Allow running directly: python fetch_data.py
if __name__ == "__main__":
    populate_multiple_symbols()
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("mode", nargs="?", default="all", help="'all' = fetch all timeframes, '1h' = only 1h")
#     args = parser.parse_args()
    
#     if args.mode == "all":
#         #populate_all_data()
#         populate_multiple_symbols()
#     else:
#         print("Run: python fetch_data.py all")

# import ccxt
# import pandas as pd
# from datetime import datetime, timedelta
# import numpy as np
# import os
# from dotenv import load_dotenv
# import time

# # Load environment variables
# load_dotenv()

# def get_db_config():
#     connection_type = os.getenv("DB_CONNECTION", "postgresql").lower()

#     if connection_type == "mysql":
#         return {
#             "type": "mysql",
#             "host": os.getenv("MYSQL_HOST", "localhost"),
#             "port": int(os.getenv("MYSQL_PORT", 3306)),
#             "database": os.getenv("MYSQL_DATABASE", "joai_db"),
#             "user": os.getenv("MYSQL_USER", "root"),
#             "password": os.getenv("MYSQL_PASSWORD", ""),
#             "charset": os.getenv("MYSQL_CHARSET", "utf8mb4")
#         }
#     elif connection_type == "postgresql":
#         database_url = os.getenv("DATABASE_URL")
#         if database_url:
#             return {
#                 "type": "postgresql",
#                 "connection_string": database_url
#             }
#         else:
#             return {
#                 "type": "postgresql",
#                 "host": os.getenv("POSTGRES_HOST", "localhost"),
#                 "port": int(os.getenv("POSTGRES_PORT", 5432)),
#                 "database": os.getenv("POSTGRES_DATABASE", "joai_db"),
#                 "user": os.getenv("POSTGRES_USER", "postgres"),
#                 "password": os.getenv("POSTGRES_PASSWORD", "")
#             }
#     else:  # default to questdb
#         from questdb.ingress import Sender, TimestampNanos
#         return {
#             "type": "questdb",
#             "url": os.getenv("QUESTDB_URL", "http://localhost:9000")
#         }

# # def fetch_candles_from_binance(symbol="BTC/USDT", timeframe="1h", limit=1000):
# #     """Fetch OHLCV data from Binance"""
# #     import logging
# #     logger = logging.getLogger(__name__)
    
# #     try:
# #         logger.info(f"Initializing Binance exchange...")
# #         exchange = ccxt.binance({
# #             'enableRateLimit': True,
# #             'timeout': 30000,  # 30 seconds timeout
# #         })
        
# #         logger.info(f"Fetching {limit} candles for {symbol} from Binance...")
# #         since = int((datetime.now() - timedelta(hours=limit)).timestamp() * 1000)
        
# #         logger.info(f"Making API call to Binance (since: {datetime.fromtimestamp(since/1000)})")
# #         candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        
# #         logger.info(f"Received {len(candles)} candles from Binance")

# #         df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
# #         df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
# #         df["symbol"] = symbol.replace("/", "")  # e.g., BTCUSDT

# #         logger.info(f"✅ Successfully fetched {len(df)} candles for {symbol}")
# #         logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
# #         return df
        
# #     except Exception as e:
# #         logger.error(f"❌ Error fetching data from Binance for {symbol}: {str(e)}")
# #         import traceback
# #         logger.error(traceback.format_exc())
# #         raise

# def fetch_candles_from_binance(symbol="BTC/USDT", timeframe="1h", limit=1000):
#     """Fetch REAL, RECENT OHLCV data from Binance — no fake old data"""
#     import ccxt
#     import time
#     import logging
#     logger = logging.getLogger(__name__)

#     exchange = ccxt.binance({
#         'enableRateLimit': True,
#         'options': {
#             'defaultType': 'spot',
#             'adjustForTimeDifference': True,
#         },
#         'timeout': 30000,
#     })

#     # CRITICAL: Do NOT set 'since' — let Binance give latest first
#     all_candles = []
#     timeframe_duration = exchange.parse_timeframe(timeframe) * 1000
#     now = exchange.milliseconds()

#     logger.info(f"Fetching {limit} {timeframe} candles for {symbol} (latest first)...")

#     while len(all_candles) < limit:
#         try:
#             candles = exchange.fetch_ohlcv(symbol, timeframe, limit=min(1000, limit - len(all_candles)))
            
#             if not candles:
#                 break
                
#             all_candles.extend(candles)
#             logger.info(f"Fetched {len(candles)} candles, total: {len(all_candles)}")

#             # Stop if we're getting old data
#             oldest_ts = candles[0][0]
#             if len(all_candles) >= limit:
#                 break

#             # Sleep to avoid rate limit
#             time.sleep(0.2)

#         except Exception as e:
#             logger.error(f"Error during fetch: {e}")
#             time.sleep(1)
#             continue

#     # Sort by timestamp and take latest N
#     all_candles.sort(key=lambda x: x[0])
#     latest_candles = all_candles[-limit:]

#     df = pd.DataFrame(latest_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
#     df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
#     df["symbol"] = symbol.replace("/", "")

#     # FINAL SANITY CHECK
#     latest_close = df['close'].iloc[-1]
#     symbol = df['symbol'].iloc[-1]
    
#     price_ranges = {
#         'BTCUSDT': (50000, 200000),
#         'ETHUSDT': (1500, 8000),
#         'BNBUSDT': (400, 1500),
#         'SOLUSDT': (80, 500),
#         'XRPUSDT': (0.3, 5),
#         'ADAUSDT': (0.2, 3),
#         'DOGEUSDT': (0.05, 1),
#         'TONUSDT': (2, 20),
#         'AVAXUSDT': (15, 200),
#         'LINKUSDT': (8, 100),
#         'SHIBUSDT': (0.000005, 0.0001),
#         'PEPEUSDT': (0.000001, 0.00005),
#     }
    
#     expected_min, expected_max = price_ranges.get(symbol, (0.001, 1000000))  # fallback
    
#     if not (expected_min <= latest_close <= expected_max):
#         print(f"WARNING: Suspicious price for {symbol}: ${latest_close:,.8f} — but allowing (might be real move)")
#         # Don't raise — just warn
#     else:
#         print(f"Price check OK → {symbol}: ${latest_close:,.6f}")

#     return df

# def fetch_candles_from_coingecko(symbol="BTC/USDT", timeframe="1h", limit=1000):
#     """Fetch data from CoinGecko API (no geo-restrictions)"""
#     import logging
#     import requests
#     logger = logging.getLogger(__name__)
    
#     try:
#         # Map symbols to CoinGecko IDs
#         symbol_map = {
#             "BTC/USDT": "bitcoin",
#             "ETH/USDT": "ethereum",
#             "BNB/USDT": "binancecoin",
#             "ADA/USDT": "cardano",
#             "SOL/USDT": "solana"
#         }
        
#         coin_id = symbol_map.get(symbol, "bitcoin")
#         days = limit // 24  # Convert hours to days
        
#         logger.info(f"Fetching {days} days of data for {coin_id} from CoinGecko...")
        
#         url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
#         params = {
#             "vs_currency": "usd",
#             "days": days,
#             "interval": "hourly"
#         }
        
#         response = requests.get(url, params=params, timeout=30)
#         response.raise_for_status()
#         data = response.json()
        
#         # Convert to DataFrame
#         prices = data['prices']
#         df = pd.DataFrame(prices, columns=['timestamp', 'close'])
#         df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
#         # CoinGecko only gives price, so we'll approximate OHLCV
#         df['open'] = df['close']
#         df['high'] = df['close'] * 1.001  # Approximate
#         df['low'] = df['close'] * 0.999   # Approximate
#         df['volume'] = 0  # CoinGecko free API doesn't provide volume
#         df['symbol'] = symbol.replace("/", "")
        
#         logger.info(f"✅ Successfully fetched {len(df)} candles for {symbol}")
#         return df
        
#     except Exception as e:
#         logger.error(f"❌ Error fetching data from CoinGecko: {str(e)}")
#         raise

# def store_candles_mysql(df, db_config):
#     """Store candles in MySQL database"""
#     try:
#         connection = pymysql.connect(
#             host=db_config["host"],
#             user=db_config["user"],
#             password=db_config["password"],
#             database=db_config["database"],
#             charset=db_config["charset"],
#             cursorclass=pymysql.cursors.DictCursor
#         )

#         try:
#             with connection.cursor() as cursor:
#                 # Insert data with ON DUPLICATE KEY UPDATE to handle duplicates
#                 sql = """
#                 INSERT INTO crypto_candles (symbol, timestamp, open, high, low, close, volume)
#                 VALUES (%s, %s, %s, %s, %s, %s, %s)
#                 ON DUPLICATE KEY UPDATE
#                 open = VALUES(open),
#                 high = VALUES(high),
#                 low = VALUES(low),
#                 close = VALUES(close),
#                 volume = VALUES(volume)
#                 """

#                 inserted_count = 0
#                 for _, row in df.iterrows():
#                     cursor.execute(sql, (
#                         row['symbol'],
#                         row['timestamp'],
#                         float(row['open']),
#                         float(row['high']),
#                         float(row['low']),
#                         float(row['close']),
#                         float(row['volume'])
#                     ))
#                     inserted_count += 1

#                 connection.commit()
#                 print(f"Stored {inserted_count} candles in MySQL database")

#         finally:
#             connection.close()

#     except Exception as e:
#         print(f"Error storing data in MySQL: {str(e)}")
#         raise

# def store_candles_questdb(df, db_config):
#     """Store candles in QuestDB database"""
#     try:
#         from questdb.ingress import Sender, TimestampNanos

#         questdb_url = db_config["url"]
#         conf = f'http::addr={questdb_url.replace("http://", "").replace("https://", "")};'

#         with Sender.from_conf(conf) as sender:
#             for _, row in df.iterrows():
#                 ts_nanos = int(row['timestamp'].timestamp() * 1_000_000_000)
#                 sender.row(
#                     'crypto_candles',
#                     symbols={'symbol': row['symbol']},
#                     columns={
#                         'open': float(row['open']),
#                         'high': float(row['high']),
#                         'low': float(row['low']),
#                         'close': float(row['close']),
#                         'volume': float(row['volume'])
#                     },
#                     at=TimestampNanos(ts_nanos)
#                 )
#             sender.flush()
#         print(f"Stored {len(df)} candles in QuestDB database")
#     except Exception as e:
#         print(f"Error storing data in QuestDB: {str(e)}")
#         raise

# def store_candles_postgresql(df, db_config):
#     """Store candles in PostgreSQL database"""
#     try:
#         import psycopg2
        
#         # Add SSL mode for external connections
#         if "connection_string" in db_config:
#             # Add sslmode=require for external Render connections
#             connection_string = db_config["connection_string"]
#             if "sslmode=" not in connection_string:
#                 connection_string += "?sslmode=require"
            
#             print(f"Connecting to PostgreSQL (external)...")
#             connection = psycopg2.connect(connection_string, connect_timeout=10)
#         else:
#             print(f"Connecting to PostgreSQL (parameters)...")
#             connection = psycopg2.connect(
#                 host=db_config["host"],
#                 user=db_config["user"],
#                 password=db_config["password"],
#                 database=db_config["database"],
#                 port=db_config["port"],
#                 connect_timeout=10,
#                 sslmode='require'  # Add SSL mode
#             )

#         print(f"✅ Connected to PostgreSQL successfully")

#         try:
#             with connection.cursor() as cursor:
#                 print(f"Inserting {len(df)} candles...")
                
#                 # Insert data with ON CONFLICT DO UPDATE
#                 sql = """
#                 INSERT INTO crypto_candles (symbol, timestamp, open, high, low, close, volume)
#                 VALUES (%s, %s, %s, %s, %s, %s, %s)
#                 ON CONFLICT (symbol, timestamp)
#                 DO UPDATE SET
#                     open = EXCLUDED.open,
#                     high = EXCLUDED.high,
#                     low = EXCLUDED.low,
#                     close = EXCLUDED.close,
#                     volume = EXCLUDED.volume
#                 """

#                 inserted_count = 0
#                 for _, row in df.iterrows():
#                     cursor.execute(sql, (
#                         row['symbol'],
#                         row['timestamp'],
#                         float(row['open']),
#                         float(row['high']),
#                         float(row['low']),
#                         float(row['close']),
#                         float(row['volume'])
#                     ))
#                     inserted_count += 1
                    
#                     # Progress indicator every 100 rows
#                     if inserted_count % 100 == 0:
#                         print(f"  Inserted {inserted_count}/{len(df)} rows...")

#                 connection.commit()
#                 print(f"✅ Stored {inserted_count} candles in PostgreSQL database")

#         finally:
#             connection.close()
#             print(f"Database connection closed")

#     except Exception as e:
#         print(f"❌ Error storing data in PostgreSQL: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         raise
    
# #uses binance
# def fetch_and_store_candles(symbol="BTC/USDT", timeframe="1h", limit=1000):
#     """Main function to fetch from Binance and store in configured database"""
#     try:
#         # Get database configuration
#         db_config = get_db_config()
#         print(f"Using database: {db_config['type']}")

#         # Fetch data from Binance
#         df = fetch_candles_from_binance(symbol, timeframe, limit)

#         # Store based on database type
#         if db_config["type"] == "mysql":
#             store_candles_mysql(df, db_config)
#         elif db_config["type"] == "postgresql":
#             store_candles_postgresql(df, db_config)
#         elif db_config["type"] == "questdb":
#             store_candles_questdb(df, db_config)
#         else:
#             raise ValueError(f"Unsupported database type: {db_config['type']}")

#         print(f"Successfully stored {len(df)} candles for {symbol}")

#     except Exception as e:
#         print(f"Error in fetch_and_store_candles: {str(e)}")
#         raise

# def populate_multiple_symbols():
#     """Populate database with data for multiple cryptocurrency symbols"""
#     import logging
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
    
#     symbols = [
#         "BTC/USDT",
#         "ETH/USDT",
#         "BNB/USDT",
#         "ADA/USDT",
#         "SOL/USDT",
#         "XRP/USDT",
#         "DOGE/USDT",
#         "SHIB/USDT",
#         "PEPE/USDT",
#         "LINK/USDT",
#         "AVAX/USDT",
#         "TON/USDT"
#     ]

#     success_count = 0
#     failed_symbols = []

#     for symbol in symbols:
#         try:
#             logger.info(f"\n{'='*50}")
#             logger.info(f"Processing {symbol}")
#             logger.info(f"{'='*50}")
            
#             fetch_and_store_candles(symbol=symbol, timeframe="1h", limit=2000)
#             success_count += 1
#             logger.info(f"✅ Successfully processed {symbol}")
            
#         except Exception as e:
#             logger.error(f"❌ Failed to process {symbol}: {str(e)}")
#             import traceback
#             logger.error(traceback.format_exc())
#             failed_symbols.append(symbol)
#             continue

#     logger.info(f"\n{'='*50}")
#     logger.info(f"SUMMARY: {success_count}/{len(symbols)} symbols processed successfully")
#     if failed_symbols:
#         logger.error(f"Failed symbols: {', '.join(failed_symbols)}")
#     logger.info(f"{'='*50}")
    
#     return {
#         "success_count": success_count,
#         "total_symbols": len(symbols),
#         "failed_symbols": failed_symbols
#     }


# import sys
# import argparse

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Fetch crypto candle data from Binance")
#     parser.add_argument("command", nargs="?", default=None, help="Type 'multi' to fetch all 12 coins")
#     parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Symbol like BTC/USDT, PEPE/USDT, etc.")
#     parser.add_argument("--limit", type=int, default=2000, help="Number of hourly candles (default: 2000)")
#     parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe: 1m, 5m, 1h, 4h, 1d etc.")

#     args = parser.parse_args()

#     # If user explicitly says "multi" → fetch all
#     if args.command == "multi" or (len(sys.argv) > 1 and sys.argv[1] == "multi"):
#         print("Fetching ALL 12 coins (2000 candles each)...")
#         populate_multiple_symbols()

#     # Otherwise → fetch single symbol mode
#     else:
#         print(f"Fetching {args.symbol} → {args.limit} candles @ {args.timeframe}")
#         fetch_and_store_candles(
#             symbol=args.symbol,
#             timeframe=args.timeframe,
#             limit=args.limit
#         )
        
#         # if __name__ == "__main__":
# #     if len(sys.argv) > 1 and sys.argv[1] == "multi":
# #         # Populate multiple symbols
# #         populate_multiple_symbols()
# #     else:
# #         # Fetch data for BTC/USDT by default
# #         fetch_and_store_candles()