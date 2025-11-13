import ccxt
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_db_config():
    connection_type = os.getenv("DB_CONNECTION", "postgresql").lower()

    if connection_type == "mysql":
        return {
            "type": "mysql",
            "host": os.getenv("MYSQL_HOST", "localhost"),
            "port": int(os.getenv("MYSQL_PORT", 3306)),
            "database": os.getenv("MYSQL_DATABASE", "joai_db"),
            "user": os.getenv("MYSQL_USER", "root"),
            "password": os.getenv("MYSQL_PASSWORD", ""),
            "charset": os.getenv("MYSQL_CHARSET", "utf8mb4")
        }
    elif connection_type == "postgresql":
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            return {
                "type": "postgresql",
                "connection_string": database_url
            }
        else:
            return {
                "type": "postgresql",
                "host": os.getenv("POSTGRES_HOST", "localhost"),
                "port": int(os.getenv("POSTGRES_PORT", 5432)),
                "database": os.getenv("POSTGRES_DATABASE", "joai_db"),
                "user": os.getenv("POSTGRES_USER", "postgres"),
                "password": os.getenv("POSTGRES_PASSWORD", "")
            }
    else:  # default to questdb
        from questdb.ingress import Sender, TimestampNanos
        return {
            "type": "questdb",
            "url": os.getenv("QUESTDB_URL", "http://localhost:9000")
        }

def fetch_candles_from_binance(symbol="BTC/USDT", timeframe="1h", limit=1000):
    """Fetch OHLCV data from Binance"""
    try:
        print(f"Fetching {limit} candles for {symbol} from Binance...")
        exchange = ccxt.binance()
        since = int((datetime.now() - timedelta(hours=limit)).timestamp() * 1000)
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["symbol"] = symbol.replace("/", "")  # e.g., BTCUSDT

        print(f"Fetched {len(df)} candles for {symbol}")
        return df
    except Exception as e:
        print(f"Error fetching data from Binance: {str(e)}")
        raise

def store_candles_mysql(df, db_config):
    """Store candles in MySQL database"""
    try:
        connection = pymysql.connect(
            host=db_config["host"],
            user=db_config["user"],
            password=db_config["password"],
            database=db_config["database"],
            charset=db_config["charset"],
            cursorclass=pymysql.cursors.DictCursor
        )

        try:
            with connection.cursor() as cursor:
                # Insert data with ON DUPLICATE KEY UPDATE to handle duplicates
                sql = """
                INSERT INTO crypto_candles (symbol, timestamp, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                open = VALUES(open),
                high = VALUES(high),
                low = VALUES(low),
                close = VALUES(close),
                volume = VALUES(volume)
                """

                inserted_count = 0
                for _, row in df.iterrows():
                    cursor.execute(sql, (
                        row['symbol'],
                        row['timestamp'],
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        float(row['volume'])
                    ))
                    inserted_count += 1

                connection.commit()
                print(f"Stored {inserted_count} candles in MySQL database")

        finally:
            connection.close()

    except Exception as e:
        print(f"Error storing data in MySQL: {str(e)}")
        raise

def store_candles_questdb(df, db_config):
    """Store candles in QuestDB database"""
    try:
        from questdb.ingress import Sender, TimestampNanos

        questdb_url = db_config["url"]
        conf = f'http::addr={questdb_url.replace("http://", "").replace("https://", "")};'

        with Sender.from_conf(conf) as sender:
            for _, row in df.iterrows():
                ts_nanos = int(row['timestamp'].timestamp() * 1_000_000_000)
                sender.row(
                    'crypto_candles',
                    symbols={'symbol': row['symbol']},
                    columns={
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume'])
                    },
                    at=TimestampNanos(ts_nanos)
                )
            sender.flush()
        print(f"Stored {len(df)} candles in QuestDB database")
    except Exception as e:
        print(f"Error storing data in QuestDB: {str(e)}")
        raise

def store_candles_postgresql(df, db_config):
    """Store candles in PostgreSQL database"""
    try:
        import psycopg2

        if "connection_string" in db_config:
            connection = psycopg2.connect(db_config["connection_string"], connect_timeout=10)
        else:
            connection = psycopg2.connect(
                host=db_config["host"],
                user=db_config["user"],
                password=db_config["password"],
                database=db_config["database"],
                port=db_config["port"],
                connect_timeout=10
            )

        try:
            with connection.cursor() as cursor:
                # Insert data with ON CONFLICT DO UPDATE (PostgreSQL equivalent of INSERT ... ON DUPLICATE KEY UPDATE)
                sql = """
                INSERT INTO crypto_candles (symbol, timestamp, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, timestamp)
                DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
                """

                inserted_count = 0
                for _, row in df.iterrows():
                    cursor.execute(sql, (
                        row['symbol'],
                        row['timestamp'],
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        float(row['volume'])
                    ))
                    inserted_count += 1

                connection.commit()
                print(f"Stored {inserted_count} candles in PostgreSQL database")

        finally:
            connection.close()

    except Exception as e:
        print(f"Error storing data in PostgreSQL: {str(e)}")
        raise
    
def fetch_and_store_candles(symbol="BTC/USDT", timeframe="1h", limit=1000):
    """Main function to fetch from Binance and store in configured database"""
    try:
        # Get database configuration
        db_config = get_db_config()
        print(f"Using database: {db_config['type']}")

        # Fetch data from Binance
        df = fetch_candles_from_binance(symbol, timeframe, limit)

        # Store based on database type
        if db_config["type"] == "mysql":
            store_candles_mysql(df, db_config)
        elif db_config["type"] == "postgresql":
            store_candles_postgresql(df, db_config)
        elif db_config["type"] == "questdb":
            store_candles_questdb(df, db_config)
        else:
            raise ValueError(f"Unsupported database type: {db_config['type']}")

        print(f"Successfully stored {len(df)} candles for {symbol}")

    except Exception as e:
        print(f"Error in fetch_and_store_candles: {str(e)}")
        raise

def populate_multiple_symbols():
    """Populate database with data for multiple cryptocurrency symbols"""
    symbols = [
        "BTC/USDT",
        "ETH/USDT",
        "BNB/USDT",
        "ADA/USDT",
        "SOL/USDT"
    ]

    for symbol in symbols:
        try:
            print(f"\n--- Processing {symbol} ---")
            fetch_and_store_candles(symbol=symbol, timeframe="1h", limit=2000)
        except Exception as e:
            print(f"Failed to process {symbol}: {e}")
            continue

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "multi":
        # Populate multiple symbols
        populate_multiple_symbols()
    else:
        # Fetch data for BTC/USDT by default
        fetch_and_store_candles()