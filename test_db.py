import pymysql
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_mysql_connection():
    """Test MySQL connection and show database statistics"""
    try:
        conn = pymysql.connect(
            host=os.getenv('MYSQL_HOST'),
            user=os.getenv('MYSQL_USER'),
            password=os.getenv('MYSQL_PASSWORD'),
            database=os.getenv('MYSQL_DATABASE')
        )

        cursor = conn.cursor()

        # Get total count
        cursor.execute('SELECT COUNT(*) FROM crypto_candles')
        total_count = cursor.fetchone()[0]
        print(f'Total candles in database: {total_count}')

        # Get symbols and their counts
        cursor.execute('SELECT symbol, COUNT(*) as count FROM crypto_candles GROUP BY symbol ORDER BY symbol')
        symbols = cursor.fetchall()
        print('Symbols in database:')
        for symbol, count in symbols:
            print(f'  {symbol}: {count} candles')

        # Get latest entries
        cursor.execute('SELECT symbol, timestamp, open, high, low, close, volume FROM crypto_candles ORDER BY timestamp DESC LIMIT 5')
        latest = cursor.fetchall()
        print('\nLatest 5 entries:')
        for row in latest:
            print(f'  {row[0]} | {row[1]} | O:{row[2]} H:{row[3]} L:{row[4]} C:{row[5]} V:{row[6]}')

        # Check api_logs table
        cursor.execute('SELECT COUNT(*) FROM api_logs')
        api_logs_count = cursor.fetchone()[0]
        print(f'\nAPI logs in database: {api_logs_count}')

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_mysql_connection()