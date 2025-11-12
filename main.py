from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
import pandas as pd
import numpy as np
from models.lstm_model import predict_next_candle
from nlp_parser import CryptoPredictionNLP
from datetime import datetime
from typing import List, Dict
import json
import logging
from questdb.ingress import Sender, TimestampNanos
import os
from dotenv import load_dotenv
import psycopg2

# Load environment variables from appropriate .env file
# Check system environment first (for Render deployment), then local .env files
app_env = os.getenv('APP_ENV')  # Check system env var first (set by Render)
# if app_env == 'production':
#     load_dotenv(dotenv_path='.env.production')
# else:
load_dotenv(dotenv_path='.env')  # Default to development

# Database configuration
def get_db_config():
    connection_type = os.getenv("DB_CONNECTION", "questdb").lower()

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
        return {
            "type": "questdb",
            "url": os.getenv("QUESTDB_URL", "http://localhost:9000")
        }

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=os.getenv("APP_NAME", "Jo AI - Crypto Prediction Engine"),
    description="Crypto prediction API with LSTM models and NLP processing"
)

class PredictRequest(BaseModel):
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"

class NLPRequest(BaseModel):
    query: str
    user_id: str = "unknown"

# Global request log storage
request_logs = []

def log_request_to_questdb(endpoint: str, request_data: dict, response_data: dict, client_ip: str = "unknown", user_id: str = "unknown", status_code: int = 200):
    try:
        db_config = get_db_config()
        if db_config["type"] == "questdb":
            questdb_url = db_config["url"]
            with Sender.from_conf(f'http::addr={questdb_url.replace("http://", "").replace("https://", "")};') as sender:
                sender.row(
                    'api_logs',
                    symbols={
                        'client_ip': client_ip or 'unknown',
                        'user_id': user_id or 'unknown',
                        'endpoint': endpoint
                    },
                    columns={
                        'request_json': json.dumps(request_data),
                        'response_json': json.dumps(response_data),
                        'status_code': status_code
                    },
                    at=TimestampNanos.now()
                )
                sender.flush()
            print(f"QuestDB Logged: {endpoint}")
        elif db_config["type"] == "mysql":
            # MySQL logging implementation
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
                    sql = """
                    INSERT INTO api_logs (client_ip, user_id, endpoint, request_json, response_json, status_code, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    """
                    cursor.execute(sql, (
                        client_ip or 'unknown',
                        user_id or 'unknown',
                        endpoint,
                        json.dumps(request_data),
                        json.dumps(response_data),
                        status_code
                    ))
                connection.commit()
                print(f"MySQL Logged: {endpoint}")
            finally:
                connection.close()
        elif db_config["type"] == "postgresql":
            # PostgreSQL logging implementation
            try:
                if "connection_string" in db_config:
                    connection = psycopg2.connect(db_config["connection_string"])
                else:
                    connection = psycopg2.connect(
                        host=db_config["host"],
                        user=db_config["user"],
                        password=db_config["password"],
                        database=db_config["database"],
                        port=db_config["port"]
                    )
                with connection.cursor() as cursor:
                    sql = """
                    INSERT INTO api_logs (client_ip, user_id, endpoint, request_json, response_json, status_code)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(sql, (
                        client_ip or 'unknown',
                        user_id or 'unknown',
                        endpoint,
                        json.dumps(request_data),
                        json.dumps(response_data),
                        status_code
                    ))
                connection.commit()
                print(f"PostgreSQL Logged: {endpoint}")
            except psycopg2.Error as e:
                print(f"PostgreSQL logging error: {e}")
            finally:
                if 'connection' in locals():
                    connection.close()
        else:
            print(f"Unsupported database type: {db_config['type']}")
    except Exception as e:
        print(f"Logging failed: {e}")
        import traceback
        traceback.print_exc()
        
def log_request(endpoint: str, request_data: dict, response_data: dict, client_ip: str = "unknown", user_id: str = "unknown"):
    """Log API requests and responses (dual logging: memory + QuestDB)"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "client_ip": client_ip,
        "user_id": user_id,
        "endpoint": endpoint,
        "request": request_data,
        "response": response_data
    }

    # Keep in memory for web display (last 100)
    #request_logs.append(log_entry)
    if len(request_logs) > 100:
        request_logs.pop(0)

    # Also save to QuestDB for persistence
    status_code = 200  # Default, could be passed as parameter
    if isinstance(response_data, dict) and "success" in response_data and response_data["success"] is False:
        status_code = 400  # Error responses

    log_request_to_questdb(endpoint, request_data, response_data, client_ip, user_id, status_code)

@app.get("/")
def home():
    """Serve the main HTML page"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content, media_type="text/html")
    except FileNotFoundError:
        return {"message": "Jo AI is running. Use POST /predict or POST /nlp_predict"}

@app.get("/test")
def test_endpoint():
    """Simple test endpoint that returns a static string"""
    return "Hello from JoAI! Server is running successfully."
@app.get("/test_db")
def test_database_connection():
    """Test PostgreSQL database connection"""
    try:
        print("Testing database connection...")
        #db_config = get_db_config()

        #if db_config["type"] == "postgresql":
            # Use hardcoded DATABASE_URL as requested
        #database_url = "postgresql://joai_user:xYl0e8Tlmz7ElkXu7w2H7m0jzIAducm8@dpg-d47pj5chg0os73frtvsg-a.oregon-postgres.render.com/joai_db"
        #connection = psycopg2.connect("postgresql://joai_user1:1Rim6AMqEOb0ZlApcFh8Ujrwt0ximlWN@dpg-d4acsls9c44c73e79or0-a.oregon-postgres.render.com/joai_db1")
        connection = psycopg2.connect(host='dpg-d4acsls9c44c73e79or0-a', dbname='joai_db1', user='joai_user1', password='1Rim6AMqEOb0ZlApcFh8Ujrwt0ximlWN', port='5432')
        cur=connection.cursor()
        cur.execute("SELECT 1 as test")
        cur.close
        connection.close()
        #connection = psycopg2.connect(database_url)

        # with connection.cursor() as cursor:
        #     cursor.execute("SELECT 1 as test")
        #     test_result = cursor.fetchone()
        #     print(f"Query result: {test_result}")

        #     # Get first record from api_logs table
        #     cursor.execute("SELECT * FROM CRYPTO_CANDLES ORDER BY TIMESTAMP DESC LIMIT 1")
        #     first_log = cursor.fetchone()
        #     print(f"First log record: {first_log}")

        connection.close()
        print("Database connection successful!")
        return {
            "connected": True,
            "message": "Database connection successful"
            # ,
            # "test_result": test_result,
            # "first_log": first_log
        }
        # else:
        #     return {"connected": False, "message": f"Database type {db_config['type']} not supported for test_db endpoint"}

    except Exception as e:
        print(f"Database connection failed: {e}")
        import traceback
        traceback.print_exc()
        return {"connected": False, "message": f"Database connection failed: {str(e)}"}

# Remove NLP processor initialization - not needed anymore

@app.post("/nlp_predict")
def nlp_predict_endpoint(request: NLPRequest, req: Request):
    """NLP prediction endpoint using real NLP parser"""
    try:
        query = request.query
        user_id = request.user_id

        print(f"NLP request: '{query}' by user {user_id}")

        # Initialize NLP processor
        from nlp_parser import CryptoPredictionNLP
        nlp_processor = CryptoPredictionNLP()

        # Process the query
        result = nlp_processor.process_query(query)

        # Log the request
        log_request(
            endpoint="/nlp_predict",
            request_data={"query": query, "user_id": user_id},
            response_data=result,
            client_ip=req.client.host if req.client else "unknown",
            user_id=user_id
        )

        return result

    except Exception as e:
        print(f"NLP endpoint error: {e}")
        import traceback
        traceback.print_exc()

        error_response = {
            "success": False,
            "message": f"Error processing NLP query: {str(e)}"
        }

        # Log error
        log_request(
            endpoint="/nlp_predict",
            request_data={"query": request.query, "user_id": request.user_id},
            response_data=error_response,
            client_ip=req.client.host if req.client else "unknown",
            user_id=request.user_id
        )

        return error_response
@app.post("/predict")
def predict_candle(request: PredictRequest, req: Request):
    print(f"Predict request received: symbol={request.symbol}, timeframe={request.timeframe}")

    try:
        # Use LSTM model for real predictions
        print("Attempting LSTM prediction...")
        prediction = predict_next_candle(request.symbol)
        print(f"LSTM prediction successful: {prediction}")

        response_data = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "prediction": {
                "open": round(prediction["open"], 2),
                "high": round(prediction["high"], 2),
                "low": round(prediction["low"], 2),
                "close": round(prediction["close"], 2),
                "volume": round(prediction["volume"], 2)
            },
            "source": "MySQL + LSTM Model + Technical Indicators"
        }

        # Extract user_id from headers (default to unknown if not provided)
        user_id = req.headers.get("X-User-ID", "unknown")

        # Log the request/response
        log_request(
            endpoint="/predict",
            request_data={"symbol": request.symbol, "timeframe": request.timeframe},
            response_data=response_data,
            client_ip=req.client.host if req.client else "unknown",
            user_id=user_id
        )

        return response_data

    except Exception as e:
        print(f"LSTM prediction failed: {e}. Using fallback model.")
        import traceback
        traceback.print_exc()
        # Fallback to simple model if LSTM fails
        print(f"LSTM prediction failed: {e}. Using fallback model.")

        # Query database for fallback
        db_config = get_db_config()

        if db_config["type"] == "mysql":
            # Query MySQL for fallback
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
                    cursor.execute(f"""
                    SELECT timestamp, open, high, low, close, volume
                    FROM crypto_candles
                    WHERE symbol = '{request.symbol}'
                    ORDER BY timestamp DESC
                    LIMIT 60
                    """)
                    data = cursor.fetchall()
            finally:
                connection.close()

            if len(data) == 0:
                raise HTTPException(status_code=404, detail="No data found for symbol")

            df = pd.DataFrame(data)
            df = df.sort_values('timestamp')

        elif db_config["type"] == "postgresql":
            # Query PostgreSQL for fallback
            try:
                if "connection_string" in db_config:
                    connection = psycopg2.connect(db_config["connection_string"])
                else:
                    connection = psycopg2.connect(
                        host=db_config["host"],
                        user=db_config["user"],
                        password=db_config["password"],
                        database=db_config["database"],
                        port=db_config["port"]
                    )
                with connection.cursor() as cursor:
                    cursor.execute(f"""
                    SELECT timestamp, open, high, low, close, volume
                    FROM crypto_candles
                    WHERE symbol = '{request.symbol}'
                    ORDER BY timestamp DESC
                    LIMIT 60
                    """)
                    data = cursor.fetchall()
                    column_names = [desc[0] for desc in cursor.description]
                    data = [dict(zip(column_names, row)) for row in data]
            except psycopg2.Error as e:
                print(f"PostgreSQL query error: {e}")
                data = []
            finally:
                if 'connection' in locals():
                    connection.close()

            if len(data) == 0:
                raise HTTPException(status_code=404, detail="No data found for symbol")

            df = pd.DataFrame(data)
            df = df.sort_values('timestamp')

        elif db_config["type"] == "questdb":
            # Query QuestDB for fallback
            query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM crypto_candles
            WHERE symbol = '{request.symbol}'
            ORDER BY timestamp DESC
            LIMIT 60
            """
            questdb_url = db_config["url"]
            response = requests.get(f"{questdb_url}/exec", params={'query': query})

            if response.status_code != 200 or 'dataset' not in response.json():
                raise HTTPException(status_code=500, detail="Failed to fetch data from QuestDB")

            data = response.json()['dataset']
            if len(data) == 0:
                raise HTTPException(status_code=404, detail="No data found for symbol")

            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = df.sort_values('timestamp')
        else:
            raise HTTPException(status_code=500, detail=f"Unsupported database type: {db_config['type']}")

        # Simple fallback prediction
        last_close = df['close'].iloc[-1]
        volatility = df['close'].pct_change().std()
        change = np.random.normal(0, volatility * 0.5)
        predicted_close = last_close * (1 + change)
        predicted_open = last_close
        predicted_high = predicted_close * (1 + abs(change) * 0.3)
        predicted_low = predicted_close * (1 - abs(change) * 0.3)

        fallback_response = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "prediction": {
                "open": round(predicted_open, 2),
                "high": round(predicted_high, 2),
                "low": round(predicted_low, 2),
                "close": round(predicted_close, 2),
                "volume": int(df['volume'].mean())
            },
            "source": "QuestDB + Fallback Model (LSTM training in progress)"
        }

        # Extract user_id from headers (default to unknown if not provided)
        user_id = req.headers.get("X-User-ID", "unknown")

        # Log the fallback response
        log_request(
            endpoint="/predict",
            request_data={"symbol": request.symbol, "timeframe": request.timeframe},
            response_data=fallback_response,
            client_ip=req.client.host if req.client else "unknown",
            user_id=user_id
        )

        return fallback_response

@app.get("/logs")
def get_logs():
    """View all API request logs"""
    return {
        "total_logs": len(request_logs),
        "logs": request_logs[-50:]  # Return last 50 logs
    }

@app.get("/logs/html")
def get_logs_html():
    """View logs from database in HTML format for easy reading - simplified version for Render"""
    print("=== /logs/html endpoint called ===")

    try:
        print("Getting database config...")
        db_config = get_db_config()
        print(f"Database type: {db_config['type']}")

        # Get logs data and count
        logs = []
        total_count = 0
        try:
            print("Attempting database connection...")
            if db_config["type"] == "postgresql":
                if "connection_string" in db_config:
                    print(f"Using connection string: {db_config['connection_string'][:50]}...")
                    connection = psycopg2.connect(db_config["connection_string"])
                else:
                    print("Using individual connection parameters")
                    connection = psycopg2.connect(
                        host=db_config["host"],
                        user=db_config["user"],
                        password=db_config["password"],
                        database=db_config["database"],
                        port=db_config["port"]
                    )
                print("Database connected successfully")
                with connection.cursor() as cursor:
                    # Get total count
                    print("Executing COUNT query...")
                    cursor.execute("SELECT COUNT(*) FROM api_logs")
                    count_result = cursor.fetchone()
                    total_count = count_result[0] if count_result else 0
                    print(f"Total count: {total_count}")

                    # Get last 50 logs
                    print("Fetching last 50 logs...")
                    cursor.execute("""
                        SELECT created_at, client_ip, user_id, endpoint, request_json, response_json, status_code
                        FROM api_logs
                        ORDER BY created_at DESC
                        LIMIT 50
                    """)
                    logs = cursor.fetchall()
                    print(f"Fetched {len(logs)} logs")

                connection.close()
                print("Database connection closed")
        except Exception as e:
            print(f"DB error: {e}")
            import traceback
            traceback.print_exc()
            total_count = f"Error: {str(e)}"
            logs = []

        html_content = f"""
        <html>
        <head>
            <title>JoAI API Request Logs ({db_config["type"].upper()})</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .log-entry {{ border: 1px solid #ccc; margin: 10px 0; padding: 10px; border-radius: 5px; }}
                .timestamp {{ color: #666; font-size: 0.9em; }}
                .endpoint {{ font-weight: bold; color: #2e7d32; }}
                .client-ip {{ color: #1976d2; }}
                .user-id {{ color: #ff9800; font-weight: bold; }}
                .request {{ background: #f5f5f5; padding: 5px; margin: 5px 0; border-radius: 3px; }}
                .response {{ background: #e8f5e8; padding: 5px; margin: 5px 0; border-radius: 3px; }}
                .error {{ background: #ffebee; color: #c62828; }}
                .success {{ border-left: 5px solid #4caf50; }}
                .error-entry {{ border-left: 5px solid #f44336; }}
                h1 {{ color: #2e7d32; }}
                .stats {{ background: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>🤖 JoAI API Request Logs ({db_config["type"].upper()})</h1>
            <div class="stats">
                <strong>Total Requests Logged:</strong> {total_count}
                <br>
                <strong>Showing Last 50:</strong> {len(logs)} entries
                <br>
                <em>Source: {db_config["type"].upper()} Database</em>
            </div>
        """

        for log in logs:
            try:
                if db_config["type"] == "questdb":
                    req_json = log[4] if len(log) > 4 else '{}'
                    resp_json = log[5] if len(log) > 5 else '{}'
                    timestamp = log[0]
                    client_ip = log[1]
                    user_id = log[2]
                    endpoint = log[3]
                elif db_config["type"] == "mysql":
                    req_json = log['request_json'] if log.get('request_json') else '{}'
                    resp_json = log['response_json'] if log.get('response_json') else '{}'
                    timestamp = log['timestamp']
                    client_ip = log['client_ip']
                    user_id = log['user_id']
                    endpoint = log['endpoint']
                elif db_config["type"] == "postgresql":
                    req_json = log[4] if len(log) > 4 else '{}'
                    resp_json = log[5] if len(log) > 5 else '{}'
                    timestamp = log[0]
                    client_ip = log[1]
                    user_id = log[2]
                    endpoint = log[3]

                # PostgreSQL JSONB columns are returned as dict objects, not strings
                req_data = req_json if isinstance(req_json, dict) else (json.loads(req_json) if req_json and req_json != 'null' else {})
                resp_data = resp_json if isinstance(resp_json, dict) else (json.loads(resp_json) if resp_json and resp_json != 'null' else {})

                status_color = "success"
                if isinstance(resp_data, dict) and resp_data.get("success") is False:
                    status_color = "error-entry"

                html_content += f"""
                <div class="log-entry {status_color}">
                    <div class="timestamp">🕐 {timestamp}</div>
                    <div class="endpoint">📍 {endpoint}</div>
                    <div class="client-ip">🌐 Client: {client_ip}</div>
                    <div class="user-id">👤 User ID: {user_id}</div>
                    <div class="request">
                        <strong>Request:</strong><br>
                        <pre>{json.dumps(req_data, indent=2)}</pre>
                    </div>
                    <div class="response">
                        <strong>Response:</strong><br>
                        <pre>{json.dumps(resp_data, indent=2)}</pre>
                    </div>
                </div>
                """
            except Exception as e:
                # Handle JSON parsing errors gracefully
                try:
                    if db_config["type"] == "postgresql":
                        timestamp = log[0] if len(log) > 0 else 'Unknown'
                        endpoint = log[3] if len(log) > 3 else 'Unknown'
                        client_ip = log[1] if len(log) > 1 else 'Unknown'
                        user_id = log[2] if len(log) > 2 else 'Unknown'
                        raw_request = str(log[4]) if len(log) > 4 else 'N/A'
                        raw_response = str(log[5]) if len(log) > 5 else 'N/A'
                    else:
                        timestamp = log.get('timestamp', log[0] if len(log) > 0 else 'Unknown')
                        endpoint = log.get('endpoint', log[3] if len(log) > 3 else 'Unknown')
                        client_ip = log.get('client_ip', log[1] if len(log) > 1 else 'Unknown')
                        user_id = log.get('user_id', log[2] if len(log) > 2 else 'Unknown')
                        raw_request = str(log.get('request_json', log[4] if len(log) > 4 else 'N/A'))
                        raw_response = str(log.get('response_json', log[5] if len(log) > 5 else 'N/A'))
                except:
                    timestamp = 'Unknown'
                    endpoint = 'Unknown'
                    client_ip = 'Unknown'
                    user_id = 'Unknown'
                    raw_request = 'Error'
                    raw_response = 'Error'

                html_content += f"""
                <div class="log-entry error">
                    <div class="timestamp">🕐 {timestamp}</div>
                    <div class="endpoint">📍 {endpoint}</div>
                    <div class="client-ip">🌐 Client: {client_ip}</div>
                    <div class="user-id">👤 User ID: {user_id}</div>
                    <div>❌ Error parsing log entry: {str(e)}</div>
                    <div class="request">
                        <strong>Raw Request:</strong><br>
                        <pre>{raw_request}</pre>
                    </div>
                    <div class="response">
                        <strong>Raw Response:</strong><br>
                        <pre>{raw_response}</pre>
                    </div>
                </div>
                """

        html_content += """
        </body>
        </html>
        """

        return HTMLResponse(content=html_content, media_type="text/html")

    except Exception as e:
        return HTMLResponse(content=f"""
        <html><body>
        <h1>❌ Error Loading Logs</h1>
        <p>Exception: {str(e)}</p>
        </body></html>
        """, media_type="text/html")

@app.delete("/logs")
def clear_logs():
    """Clear all request logs"""
    global request_logs
    request_logs.clear()
    return {"message": "All logs cleared"}

@app.post("/init_db")
def init_database():
    """Initialize database: check/create tables and populate crypto data"""
    try:
        db_config = get_db_config()

        if db_config["type"] != "postgresql":
            return {
                "success": False,
                "message": f"Database initialization only supported for PostgreSQL. Current type: {db_config['type']}"
            }

        # Import required functions
        from fetch_data import populate_multiple_symbols

        # Connect to database
        if "connection_string" in db_config:
            connection = psycopg2.connect(db_config["connection_string"])
        else:
            connection = psycopg2.connect(
                host=db_config["host"],
                user=db_config["user"],
                password=db_config["password"],
                database=db_config["database"],
                port=db_config["port"]
            )

        results = {
            "api_logs": {"status": "unknown", "action": "none"},
            "crypto_candles": {"status": "unknown", "action": "none"}
        }

        try:
            with connection.cursor() as cursor:
                # Check and create api_logs table
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = 'api_logs'
                    )
                """)
                api_logs_exists = cursor.fetchone()[0]

                if not api_logs_exists:
                    # Create api_logs table
                    create_api_logs_table = """
                    CREATE TABLE api_logs (
                        id SERIAL PRIMARY KEY,
                        client_ip VARCHAR(45) NOT NULL,
                        user_id VARCHAR(100) NOT NULL DEFAULT 'unknown',
                        endpoint VARCHAR(255) NOT NULL,
                        request_json JSONB,
                        response_json JSONB,
                        status_code INT NOT NULL DEFAULT 200,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX idx_api_logs_client_ip ON api_logs (client_ip);
                    CREATE INDEX idx_api_logs_user_id ON api_logs (user_id);
                    CREATE INDEX idx_api_logs_endpoint ON api_logs (endpoint);
                    CREATE INDEX idx_api_logs_created_at ON api_logs (created_at);
                    """
                    cursor.execute(create_api_logs_table)
                    results["api_logs"] = {"status": "created", "action": "created table with indexes"}
                    print("api_logs table created successfully")
                else:
                    # Check if table has required columns
                    cursor.execute("""
                        SELECT column_name FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = 'api_logs'
                        ORDER BY ordinal_position
                    """)
                    columns = [row[0] for row in cursor.fetchall()]
                    required_columns = ['id', 'client_ip', 'user_id', 'endpoint', 'request_json', 'response_json', 'status_code', 'created_at']

                    if all(col in columns for col in required_columns):
                        results["api_logs"] = {"status": "exists", "action": "table exists with required fields"}
                    else:
                        results["api_logs"] = {"status": "incomplete", "action": "table exists but missing some fields"}
                        print(f"api_logs table exists but missing columns. Has: {columns}")

                # Check and create/populate crypto_candles table
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = 'crypto_candles'
                    )
                """)
                crypto_candles_exists = cursor.fetchone()[0]

                if not crypto_candles_exists:
                    # Create crypto_candles table
                    create_crypto_candles_table = """
                    CREATE TABLE crypto_candles (
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
                    CREATE INDEX idx_crypto_candles_symbol ON crypto_candles (symbol);
                    CREATE INDEX idx_crypto_candles_timestamp ON crypto_candles (timestamp);
                    CREATE INDEX idx_crypto_candles_symbol_timestamp ON crypto_candles (symbol, timestamp);
                    """
                    cursor.execute(create_crypto_candles_table)
                    results["crypto_candles"] = {"status": "created", "action": "created table with indexes"}
                    print("crypto_candles table created successfully")
                    populate_data = True
                else:
                    # Check if table has data
                    cursor.execute("SELECT COUNT(*) FROM crypto_candles")
                    data_count = cursor.fetchone()[0]

                    if data_count == 0:
                        results["crypto_candles"] = {"status": "empty", "action": "table exists but no data"}
                        print(f"crypto_candles table exists but is empty (0 records)")
                        populate_data = True
                    else:
                        results["crypto_candles"] = {"status": "populated", "action": f"table exists with {data_count} records"}
                        print(f"crypto_candles table exists with {data_count} records")
                        populate_data = False

            connection.commit()

            # Populate crypto data if needed
            if populate_data:
                print("Starting data population...")
                try:
                    populate_multiple_symbols()
                    results["crypto_candles"]["action"] += " - data populated"
                    print("Data population completed successfully")
                except Exception as e:
                    results["crypto_candles"]["action"] += f" - data population failed: {str(e)}"
                    print(f"Data population failed: {str(e)}")

        finally:
            connection.close()

        return {
            "success": True,
            "message": "Database initialization completed",
            "results": results
        }

    except Exception as e:
        print(f"Database initialization error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Database initialization failed: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("APP_PORT", 8081))
    uvicorn.run(app, host="0.0.0.0", port=port)