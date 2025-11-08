import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_postgresql_config():
    """Get PostgreSQL configuration from environment or DATABASE_URL"""
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        # Parse DATABASE_URL for Render deployment
        # Format: postgresql://user:password@host:port/database
        return {"connection_string": database_url}
    else:
        # Fallback to individual environment variables
        return {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DATABASE", "joai_db"),
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", ""),
        }

def create_database_and_tables():
    """Initialize PostgreSQL database and create necessary tables"""
    config = get_postgresql_config()

    try:
        # Connect to PostgreSQL
        if "connection_string" in config:
            connection = psycopg2.connect(config["connection_string"])
        else:
            connection = psycopg2.connect(**config)

        print("Connected to PostgreSQL successfully")

        try:
            with connection.cursor() as cursor:
                # Create api_logs table with JSONB for better JSON handling
                create_api_logs_table = """
                CREATE TABLE IF NOT EXISTS api_logs (
                    id SERIAL PRIMARY KEY,
                    client_ip VARCHAR(45) NOT NULL,
                    user_id VARCHAR(100) NOT NULL DEFAULT 'unknown',
                    endpoint VARCHAR(255) NOT NULL,
                    request_json JSONB,
                    response_json JSONB,
                    status_code INT NOT NULL DEFAULT 200,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(client_ip, user_id, endpoint, created_at)
                );

                CREATE INDEX IF NOT EXISTS idx_api_logs_client_ip ON api_logs (client_ip);
                CREATE INDEX IF NOT EXISTS idx_api_logs_user_id ON api_logs (user_id);
                CREATE INDEX IF NOT EXISTS idx_api_logs_endpoint ON api_logs (endpoint);
                CREATE INDEX IF NOT EXISTS idx_api_logs_created_at ON api_logs (created_at);
                """
                cursor.execute(create_api_logs_table)
                print("api_logs table created successfully")

                # Create crypto_candles table for time-series data
                create_crypto_candles_table = """
                CREATE TABLE IF NOT EXISTS crypto_candles (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    open DECIMAL(20,8) NOT NULL,
                    high DECIMAL(20,8) NOT NULL,
                    low DECIMAL(20,8) NOT NULL,
                    close DECIMAL(20,8) NOT NULL,
                    volume DECIMAL(20,8) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                );

                CREATE INDEX IF NOT EXISTS idx_crypto_candles_symbol ON crypto_candles (symbol);
                CREATE INDEX IF NOT EXISTS idx_crypto_candles_timestamp ON crypto_candles (timestamp);
                CREATE INDEX IF NOT EXISTS idx_crypto_candles_symbol_timestamp ON crypto_candles (symbol, timestamp);
                """
                cursor.execute(create_crypto_candles_table)
                print("crypto_candles table created successfully")

            connection.commit()
            print("All tables created successfully!")

        finally:
            connection.close()

    except psycopg2.Error as e:
        print(f"PostgreSQL Error: {e}")
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    create_database_and_tables()