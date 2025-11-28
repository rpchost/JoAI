from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
import pandas as pd
import numpy as np
#from models.lstm_model import predict_next_candle
#from nlp_parser import CryptoPredictionNLP
from datetime import datetime
from typing import List, Dict, Optional
import json
import logging
from questdb.ingress import Sender, TimestampNanos
import os
from dotenv import load_dotenv
import psycopg2
from ai_analyst import ask_joai

# Load environment variables from appropriate .env file
# Check system environment first (for Render deployment), then local .env files
app_env = os.getenv('APP_ENV')  # Check system env var first (set by Render)
# if app_env == 'production':
#     load_dotenv(dotenv_path='.env.production')
# else:
#load_dotenv(dotenv_path='.env')  # Default to development

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
    symbol: str = "BTCUSD"
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
    """Test PostgreSQL database connection with detailed logging"""
    try:
        logger.info("=== Starting database connection test ===")
        
        # Check if DATABASE_URL exists
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            logger.error("DATABASE_URL environment variable not found!")
            return {
                "connected": False,
                "message": "DATABASE_URL not configured",
                "env_vars": list(os.environ.keys())  # List available env vars
            }
        
        logger.info(f"DATABASE_URL found (first 30 chars): {db_url[:30]}...")
        logger.info("Attempting to connect...")
        
        connection = psycopg2.connect(db_url, connect_timeout=10)
        logger.info("Connection established")
        
        cur = connection.cursor()
        logger.info("Cursor created")
        
        cur.execute("SELECT version(), current_database()")
        result = cur.fetchone()
        logger.info(f"Query executed successfully: {result}")
        
        cur.close()
        connection.close()
        logger.info("Connection closed")
       
        return {
            "connected": True,
            "message": "Database connection successful",
            "db_version": result[0][:50] if result else None,
            "database": result[1] if result and len(result) > 1 else None
        }
        
    except psycopg2.OperationalError as e:
        logger.error(f"PostgreSQL Operational Error: {str(e)}")
        return {
            "connected": False,
            "error_type": "OperationalError",
            "message": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return {
            "connected": False,
            "error_type": type(e).__name__,
            "message": str(e)
        }

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
        # Lazy load - only import when needed
        print("Loading LSTM model...")
        from models.lstm_model import predict_next_candle

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

@app.get("/readjoAiApiLogs")
def read_joai_api_logs(page: Optional[int] = None, limit: Optional[int] = None):
    """Read rows from api_logs table, ordered by created_at DESC. If page/limit not provided, returns all records."""
    try:
        db_config = get_db_config()
        logs = []

        if db_config["type"] == "postgresql":
            if "connection_string" in db_config:
                connection = psycopg2.connect(db_config["connection_string"])
            else:
                connection = psycopg2.connect(
                    host=db_config["host"],
                    port=db_config["port"],
                    database=db_config["database"],
                    user=db_config["user"],
                    password=db_config["password"]
                )
            with connection.cursor() as cursor:
                if page is not None and limit is not None:
                    # Paginated query
                    offset = (page - 1) * limit
                    cursor.execute("SELECT * FROM api_logs ORDER BY created_at DESC LIMIT %s OFFSET %s", (limit, offset))
                else:
                    # Return all records
                    cursor.execute("SELECT * FROM api_logs ORDER BY created_at DESC")
                rows = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]
                logs = [dict(zip(column_names, row)) for row in rows]
            connection.close()

        elif db_config["type"] == "mysql":
            import pymysql
            connection = pymysql.connect(
                host=db_config["host"],
                port=db_config["port"],
                database=db_config["database"],
                user=db_config["user"],
                password=db_config["password"],
                charset=db_config["charset"],
                cursorclass=pymysql.cursors.DictCursor
            )
            with connection.cursor() as cursor:
                if page is not None and limit is not None:
                    # Paginated query
                    offset = (page - 1) * limit
                    cursor.execute("SELECT * FROM api_logs ORDER BY created_at DESC LIMIT %s OFFSET %s", (limit, offset))
                else:
                    # Return all records
                    cursor.execute("SELECT * FROM api_logs ORDER BY created_at DESC")
                logs = cursor.fetchall()
            connection.close()

        elif db_config["type"] == "questdb":
            if page is not None and limit is not None:
                # Paginated query
                offset = (page - 1) * limit
                query = f"SELECT * FROM api_logs ORDER BY created_at DESC LIMIT {limit} OFFSET {offset}"
            else:
                # Return all records
                query = "SELECT * FROM api_logs ORDER BY created_at DESC"
            response = requests.get(f"{db_config['url']}/exec", params={'query': query})
            if response.status_code == 200 and 'dataset' in response.json():
                data = response.json()['dataset']
                columns = response.json().get('columns', [])
                logs = [dict(zip(columns, row)) for row in data]
            else:
                raise Exception("Failed to query QuestDB")

        else:
            raise HTTPException(status_code=500, detail=f"Unsupported database type: {db_config['type']}")

        return {"logs": logs}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/init_db")
def init_database():
    """Initialize database: check/create tables and populate crypto data"""
    try:
        logger.info("=== Starting database initialization ===")
        
        # Check if DATABASE_URL exists
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            logger.error("DATABASE_URL environment variable not found!")
            return {
                "success": False,
                "message": "DATABASE_URL not configured",
                "env_vars": list(os.environ.keys())
            }
        
        logger.info(f"DATABASE_URL found (first 30 chars): {db_url[:30]}...")
        
        # Import required functions
        from fetch_data import populate_multiple_symbols

        # Connect to database with timeout
        logger.info("Attempting to connect to database...")
        connection = psycopg2.connect(db_url, connect_timeout=10)
        logger.info("Connection established successfully")

        results = {
            "api_logs": {"status": "unknown", "action": "none"},
            "crypto_candles": {"status": "unknown", "action": "none"}
        }

        try:
            with connection.cursor() as cursor:
                # Check and create api_logs table
                logger.info("Checking api_logs table...")
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = 'api_logs'
                    )
                """)
                api_logs_exists = cursor.fetchone()[0]

                if not api_logs_exists:
                    # Create api_logs table
                    logger.info("Creating api_logs table...")
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
                    logger.info("api_logs table created successfully")
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
                        logger.info("api_logs table already exists with all required fields")
                    else:
                        results["api_logs"] = {"status": "incomplete", "action": "table exists but missing some fields"}
                        logger.warning(f"api_logs table exists but missing columns. Has: {columns}")

                # Check and create/populate crypto_candles table
                logger.info("Checking crypto_candles table...")
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = 'crypto_candles'
                    )
                """)
                crypto_candles_exists = cursor.fetchone()[0]

                if not crypto_candles_exists:
                    # Create crypto_candles table
                    logger.info("Creating crypto_candles table...")
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
                    logger.info("crypto_candles table created successfully")
                    populate_data = True
                else:
                    # Check if table has data
                    cursor.execute("SELECT COUNT(*) FROM crypto_candles")
                    data_count = cursor.fetchone()[0]

                    if data_count == 0:
                        results["crypto_candles"] = {"status": "empty", "action": "table exists but no data"}
                        logger.info(f"crypto_candles table exists but is empty (0 records)")
                        populate_data = True
                    else:
                        results["crypto_candles"] = {"status": "populated", "action": f"table exists with {data_count} records"}
                        logger.info(f"crypto_candles table exists with {data_count} records")
                        populate_data = False

            connection.commit()
            logger.info("Database changes committed")

            # Populate crypto data if needed
            if populate_data:
                logger.info("Starting data population...")
                try:
                    populate_multiple_symbols()
                    results["crypto_candles"]["action"] += " - data populated"
                    logger.info("Data population completed successfully")
                except Exception as e:
                    results["crypto_candles"]["action"] += f" - data population failed: {str(e)}"
                    logger.error(f"Data population failed: {str(e)}")

        finally:
            connection.close()
            logger.info("Database connection closed")

        return {
            "success": True,
            "message": "Database initialization completed",
            "results": results
        }

    except psycopg2.OperationalError as e:
        logger.error(f"PostgreSQL Operational Error: {str(e)}")
        return {
            "success": False,
            "error_type": "OperationalError",
            "message": str(e)
        }
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}", exc_info=True)
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Database initialization failed: {str(e)}"
        }
    
@app.get("/test_binance")
def test_binance_connection():
    """Test if we can connect to Binance API"""
    try:
        import ccxt
        logger.info("Initializing Binance exchange...")
        
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 30000,
        })
        
        logger.info("Fetching BTC/USDT ticker...")
        ticker = exchange.fetch_ticker('BTC/USDT')
        
        logger.info(f"Successfully fetched ticker: {ticker['last']}")
        
        return {
            "success": True,
            "message": "Successfully connected to Binance",
            "btc_price": ticker['last'],
            "timestamp": ticker['timestamp']
        }
        
    except Exception as e:
        logger.error(f"Failed to connect to Binance: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "message": f"Failed to connect to Binance: {str(e)}"
        }
    
@app.post("/ai_analyst")
def ai_analyst_endpoint(request: dict):
    """
    AI-powered Bitcoin analysis using on-chain data + LSTM predictions
    
    Example request:
    {
        "question": "Should I buy Bitcoin now?"
    }
    """
    try:
        question = request.get("question", "What's the current Bitcoin trend?")
        answer = ask_joai(question)
        
        return {
            "success": True,
            "question": question,
            "answer": answer,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/market_sentiment")
def get_market_sentiment():
    """Quick sentiment check without AI processing"""
    from ai_analyst import get_onchain_data, get_price_data
    
    try:
        price_df = get_price_data()
        onchain = get_onchain_data()
        
        if price_df.empty:
            return {"error": "No price data available"}
        
        current_price = float(price_df['close'].iloc[-1])
        
        return {
            "success": True,
            "price": current_price,
            "fear_greed": onchain.get("fear_greed", 50),
            "sentiment": onchain.get("classification", "Neutral"),
            "exchange_reserves_trend": onchain.get("trend", "unknown"),
            "mempool_tx": onchain.get("unconfirmed_tx", 0),
            "volume_24h": onchain.get("total_volume_24h", 0),
            "price_change_24h": onchain.get("price_change_24h", 0),
            "active_addresses": onchain.get("active_addresses_24h", 0)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    
@app.get("/test_coingecko")
def test_coingecko_and_populate():
    """Test CoinGecko API and populate database with BTC data"""
    try:
        logger.info("=== Testing CoinGecko API ===")
        
        # Import the function
        from fetch_data import fetch_and_store_candles
        
        # Test with BTC/USDT
        logger.info("Fetching and storing BTC/USDT data...")
        result = fetch_and_store_candles(symbol="BTC/USDT", timeframe="1h", limit=100)
        
        # Verify data was stored
        db_url = os.getenv("DATABASE_URL")
        connection = psycopg2.connect(db_url, connect_timeout=10)
        
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM crypto_candles WHERE symbol = 'BTCUSD'")
            count = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT symbol, timestamp, open, high, low, close, volume 
                FROM crypto_candles 
                WHERE symbol = 'BTCUSD' 
                ORDER BY timestamp DESC 
                LIMIT 5
            """)
            recent_candles = cursor.fetchall()
        
        connection.close()
        
        return {
            "success": True,
            "message": "CoinGecko test successful",
            "rows_inserted": result.get("rows_inserted", 0),
            "total_btc_candles": count,
            "recent_sample": [
                {
                    "symbol": row[0],
                    "timestamp": str(row[1]),
                    "open": float(row[2]),
                    "high": float(row[3]),
                    "low": float(row[4]),
                    "close": float(row[5]),
                    "volume": float(row[6])
                }
                for row in recent_candles
            ]
        }
        
    except Exception as e:
        logger.error(f"CoinGecko test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "message": f"CoinGecko test failed: {str(e)}"
        }
    
@app.post("/joai")
async def joai_master_endpoint(request: dict):
    user_question = request.get("question", "").strip()
    if not user_question:
        return {"success": False, "error": "No question provided"}

    try:
        from nlp_parser import CryptoPredictionNLP
        from datetime import datetime

        nlp = CryptoPredictionNLP()
        nlp_result = nlp.process_query(user_question)

        symbol = nlp_result.get("symbol", "BTC").upper().replace("USD", "USDT")
        timeframe = nlp_result.get("timeframe", "1 hour")
        full_prediction = nlp_result.get("prediction", {})

        # ── SAFELY extract close price as float ──
        def to_float(val, default=91450.0):
            if val is None:
                return default
            try:
                return float(str(val).replace("$", "").replace(",", "").strip())
            except:
                return default

        lstm_close = to_float(full_prediction.get("close"))
        lstm_open  = to_float(full_prediction.get("open"))
        lstm_high  = to_float(full_prediction.get("high"))
        lstm_low   = to_float(full_prediction.get("low"))
        lstm_volume = to_float(full_prediction.get("volume", 0))

        direction = "bullish" if lstm_close >= 90000 else "bearish"

        # ── Build context for Groq (100% safe formatting) ──
        rich_question = f"""
User asked: "{user_question}"

Parsed:
  Symbol: {symbol}
  Timeframe: {timeframe}
  LSTM Prediction (next {timeframe}):
    Open:   ${lstm_open:,.2f}
    High:   ${lstm_high:,.2f}
    Low:    ${lstm_low:,.2f}
    Close:  ${lstm_close:,.2f}
    Volume: {lstm_volume:,.0f}

Current sentiment: Extreme Fear (25/100), dropping exchange reserves.
Give a sharp, confident, institutional answer in 3–5 sentences. Be bold.
        """.strip()

        joai_answer = ask_joai(rich_question)

        return {
            "success": True,
            "question": user_question,
            "parsed": {
                "symbol": symbol,
                "timeframe": timeframe,
                "intent": nlp_result.get("intent", "unknown")
            },
            "lstm_prediction": {
                "open": round(lstm_open, 2),
                "high": round(lstm_high, 2),
                "low": round(lstm_low, 2),
                "close": round(lstm_close, 2),
                "volume": int(lstm_volume)
            },
            "joai_analysis": joai_answer,
            "source": "JoAI Ultimate Brain v3 – Fully Unbreakable",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": f"Server error: {str(e)}",
            "fallback_analysis": ask_joai(user_question or "What is Bitcoin doing right now?")
        }