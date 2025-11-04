import pymysql
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_mysql_config():
    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", 3306)),
        "database": os.getenv("MYSQL_DATABASE", "joai_db"),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "charset": os.getenv("MYSQL_CHARSET", "utf8mb4")
    }

def create_database_and_tables():
    """Initialize MySQL database and create necessary tables"""
    config = get_mysql_config()

    try:
        # Connect without specifying database first
        temp_config = config.copy()
        temp_config.pop('database', None)

        connection = pymysql.connect(**temp_config)
        print("Connected to MySQL server successfully")

        try:
            with connection.cursor() as cursor:
                # Create database if it doesn't exist
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                print(f"Database '{config['database']}' created or already exists")

                # Use the database
                cursor.execute(f"USE {config['database']}")

                # Create api_logs table
                create_api_logs_table = """
                CREATE TABLE IF NOT EXISTS api_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    client_ip VARCHAR(45) NOT NULL,
                    user_id VARCHAR(100) NOT NULL DEFAULT 'unknown',
                    endpoint VARCHAR(255) NOT NULL,
                    request_json JSON,
                    response_json JSON,
                    status_code INT NOT NULL DEFAULT 200,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_client_ip (client_ip),
                    INDEX idx_user_id (user_id),
                    INDEX idx_endpoint (endpoint),
                    INDEX idx_created_at (created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
                cursor.execute(create_api_logs_table)
                print("api_logs table created successfully")

                # Create crypto_candles table for fallback data (similar to QuestDB structure)
                create_crypto_candles_table = """
                CREATE TABLE IF NOT EXISTS crypto_candles (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open DECIMAL(20,8) NOT NULL,
                    high DECIMAL(20,8) NOT NULL,
                    low DECIMAL(20,8) NOT NULL,
                    close DECIMAL(20,8) NOT NULL,
                    volume DECIMAL(20,8) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_symbol_timestamp (symbol, timestamp),
                    INDEX idx_symbol (symbol),
                    INDEX idx_timestamp (timestamp)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
                cursor.execute(create_crypto_candles_table)
                print("crypto_candles table created successfully")

            connection.commit()
            print("All tables created successfully!")

        finally:
            connection.close()

    except pymysql.Error as e:
        print(f"MySQL Error: {e}")
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    create_database_and_tables()