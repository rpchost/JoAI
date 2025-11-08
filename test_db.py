import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_database_connection():
    """Test database connection and show database statistics"""
    db_type = os.getenv('DB_CONNECTION', 'questdb').lower()

    try:
        if db_type == 'postgresql':
            # Test PostgreSQL connection
            database_url = os.getenv("DATABASE_URL")
            if database_url:
                conn = psycopg2.connect(database_url)
            else:
                conn = psycopg2.connect(
                    host=os.getenv('POSTGRES_HOST', 'localhost'),
                    user=os.getenv('POSTGRES_USER', 'postgres'),
                    password=os.getenv('POSTGRES_PASSWORD', ''),
                    database=os.getenv('POSTGRES_DATABASE', 'joai_db'),
                    port=int(os.getenv('POSTGRES_PORT', 5432))
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
            print("PostgreSQL connection test completed successfully!")

        elif db_type == 'mysql':
            # Original MySQL test code
            import pymysql
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
            print("MySQL connection test completed successfully!")

        else:
            print(f"Unsupported database type: {db_type}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_database_connection()