import requests

def create_table():
    url = "http://localhost:9000/exec"

    # Create crypto_candles table (unchanged)
    query1 = """
    CREATE TABLE IF NOT EXISTS crypto_candles (
        timestamp TIMESTAMP,
        symbol SYMBOL INDEX,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume LONG
    ) TIMESTAMP(timestamp) PARTITION BY DAY;
    """
    response1 = requests.get(url, params={'query': query1, 'fmt': 'json'})
    if response1.status_code == 200:
        print("Table 'crypto_candles' created successfully.")
    else:
        print("Error creating crypto_candles:", response1.text)

    # Create api_logs table with user_id column
    query2 = """
    CREATE TABLE IF NOT EXISTS api_logs (
        timestamp TIMESTAMP,
        client_ip SYMBOL,
        user_id SYMBOL,
        endpoint SYMBOL,
        request_json STRING,
        response_json STRING,
        status_code INT
    ) timestamp(timestamp) PARTITION BY DAY;
    """
    response2 = requests.get(url, params={'query': query2, 'fmt': 'json'})
    if response2.status_code == 200:
        print("Table 'api_logs' created successfully with user_id column.")
    else:
        print("Error creating api_logs:", response2.text)

if __name__ == "__main__":
    create_table()