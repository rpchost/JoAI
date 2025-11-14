import numpy as np
import pandas as pd
import os
import pickle
import psycopg2
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# CRITICAL: Don't import TensorFlow at module level to save memory!
# Only import when actually needed for training/prediction

def get_db_config():
    connection_type = os.getenv("DB_CONNECTION", "postgresql").lower()

    if connection_type == "postgresql":
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
    else:
        raise Exception(f"Unsupported database type: {connection_type}")

class LSTMCryptoPredictor:
    # Class-level cache for model and scalers (shared across all instances)
    _model_cache = {}
    _instance_count = 0
    
    def __init__(self, symbol="BTCUSDT", sequence_length=60, model_path="models/saved_model.keras"):
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.model_path = model_path
        self.scaler_path = model_path.replace('.keras', '_scaler.pkl')
        self.target_scaler_path = model_path.replace('.keras', '_target_scaler.pkl')
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = None
        
        LSTMCryptoPredictor._instance_count += 1
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

    def fetch_training_data(self, limit=1000):
        """Fetch historical data with timeout (OPTIMIZED)"""
        db_config = get_db_config()

        if db_config["type"] == "postgresql":
            try:
                if "connection_string" in db_config:
                    connection = psycopg2.connect(
                        db_config["connection_string"],
                        connect_timeout=10  # Timeout protection
                    )
                else:
                    connection = psycopg2.connect(
                        host=db_config["host"],
                        user=db_config["user"],
                        password=db_config["password"],
                        database=db_config["database"],
                        port=db_config["port"],
                        connect_timeout=10
                    )
                    
                with connection.cursor() as cursor:
                    # Use parameterized query (SQL injection protection)
                    cursor.execute("""
                        SELECT timestamp, open, high, low, close, volume
                        FROM crypto_candles
                        WHERE symbol = %s
                        ORDER BY timestamp ASC
                        LIMIT %s
                    """, (self.symbol, limit))
                    data = cursor.fetchall()
                    column_names = [desc[0] for desc in cursor.description]
                    data = [dict(zip(column_names, row)) for row in data]
            except psycopg2.Error as e:
                raise Exception(f"PostgreSQL query error: {e}")
            finally:
                if 'connection' in locals():
                    connection.close()

            if len(data) < self.sequence_length + 10:
                raise Exception(f"Insufficient data for training. Need at least {self.sequence_length + 10} records, got {len(data)}")

            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.sort_values('timestamp').dropna()
        else:
            raise Exception(f"Unsupported database type: {db_config['type']}")

        return df

    def prepare_data(self, df):
        """Prepare data with technical indicators and scaling (OPTIMIZED)"""
        from utils.indicators import add_technical_indicators
        
        # Add technical indicators (all 22 features)
        df = add_technical_indicators(df)

        # Define feature columns (OHLCV + 17 technical indicators = 22 features)
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi',
            'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'atr',
            'stoch_k', 'stoch_d', 'willr', 'volume_sma', 'roc'
        ]

        # Validate all features exist
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        # Scale features (22 features)
        feature_data = df[self.feature_columns].values
        scaled_data = self.scaler.fit_transform(feature_data)

        # Scale targets separately (next close price)
        targets = df['close'].shift(-1).values[:-1]
        scaled_targets = self.target_scaler.fit_transform(targets.reshape(-1, 1))

        return scaled_data[:-1], scaled_targets

    def create_sequences(self, data, targets):
        """Create sequences for LSTM input (OPTIMIZED)"""
        # Pre-allocate arrays for efficiency
        n_samples = len(data) - self.sequence_length
        X = np.zeros((n_samples, self.sequence_length, data.shape[1]), dtype=np.float32)
        y = np.zeros((n_samples, 1), dtype=np.float32)
        
        for i in range(n_samples):
            X[i] = data[i:(i + self.sequence_length)]
            y[i] = targets[i + self.sequence_length]

        return X, y

    def build_model(self, input_shape):
        """Build LSTM model architecture (KEPT ORIGINAL QUALITY)"""
        # Lazy import TensorFlow only when needed
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
        from tensorflow.keras.optimizers import Adam
        
        # SAME ARCHITECTURE - High quality model
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape)),
            Dropout(0.2),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)  # Predict next close price
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )

        return model

    def train(self, epochs=100, batch_size=32, validation_split=0.2):
        """Train the LSTM model (OPTIMIZED WITH CLEANUP)"""
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        import tensorflow as tf
        
        print(f"Fetching training data for {self.symbol}...")
        df = self.fetch_training_data(limit=2000)

        print(f"Preparing data with {len(df)} records...")
        scaled_data, scaled_targets = self.prepare_data(df)

        X, y = self.create_sequences(scaled_data, scaled_targets)
        print(f"Training data shape: {X.shape}, Target shape: {y.shape}")

        # Build model
        self.model = self.build_model((X.shape[1], X.shape[2]))

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        checkpoint = ModelCheckpoint(
            self.model_path, 
            monitor='val_loss', 
            save_best_only=True
        )

        # Train model
        print("Training LSTM model...")
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )

        # Save scalers
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(self.target_scaler_path, 'wb') as f:
            pickle.dump(self.target_scaler, f)

        print(f"Model trained and saved to {self.model_path}")
        
        # CRITICAL: Clear TensorFlow session to free memory
        tf.keras.backend.clear_session()
        
        return history

    def load_model(self):
        """Load trained model with caching (OPTIMIZED)"""
        cache_key = self.model_path
        
        # Check class-level cache first
        if cache_key in self._model_cache:
            print(f"Loading model from memory cache")
            cached = self._model_cache[cache_key]
            self.model = cached['model']
            self.scaler = cached['scaler']
            self.target_scaler = cached['target_scaler']
            self.feature_columns = cached['features']
            return

        if os.path.exists(self.model_path):
            # Lazy import TensorFlow
            from tensorflow.keras.models import load_model as keras_load_model
            
            print(f"Loading model from disk: {self.model_path}")
            self.model = keras_load_model(self.model_path)

            # Load scalers
            if os.path.exists(self.scaler_path) and os.path.exists(self.target_scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                with open(self.target_scaler_path, 'rb') as f:
                    self.target_scaler = pickle.load(f)
            else:
                raise Exception("Scalers not found. Please retrain the model.")

            # Set feature columns
            self.feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi',
                'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_middle', 'bb_lower', 'atr',
                'stoch_k', 'stoch_d', 'willr', 'volume_sma', 'roc'
            ]
            
            # Cache the loaded model (shared across all instances)
            self._model_cache[cache_key] = {
                'model': self.model,
                'scaler': self.scaler,
                'target_scaler': self.target_scaler,
                'features': self.feature_columns
            }
            print(f"Model cached in memory")
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

    def predict_next_candle(self, recent_data=None):
        """Predict next candle (OPTIMIZED WITH TIMEOUTS)"""
        if self.model is None:
            try:
                self.load_model()
            except FileNotFoundError:
                raise Exception("Model not trained yet. Please train the model first.")

        # Get recent data
        if recent_data is None:
            db_config = get_db_config()

            if db_config["type"] == "postgresql":
                try:
                    if "connection_string" in db_config:
                        connection = psycopg2.connect(
                            db_config["connection_string"],
                            connect_timeout=10  # Timeout protection
                        )
                    else:
                        connection = psycopg2.connect(
                            host=db_config["host"],
                            user=db_config["user"],
                            password=db_config["password"],
                            database=db_config["database"],
                            port=db_config["port"],
                            connect_timeout=10
                        )
                        
                    with connection.cursor() as cursor:
                        cursor.execute("""
                            SELECT timestamp, open, high, low, close, volume
                            FROM crypto_candles
                            WHERE symbol = %s
                            ORDER BY timestamp DESC
                            LIMIT %s
                        """, (self.symbol, self.sequence_length + 10))
                        data = cursor.fetchall()
                        column_names = [desc[0] for desc in cursor.description]
                        data = [dict(zip(column_names, row)) for row in data]
                except psycopg2.Error as e:
                    raise Exception(f"PostgreSQL query error: {e}")
                finally:
                    if 'connection' in locals():
                        connection.close()

                if not data:
                    raise Exception(f"No data found for symbol {self.symbol}")

                recent_data = pd.DataFrame(data)
                recent_data['timestamp'] = pd.to_datetime(recent_data['timestamp'], errors='coerce')
                recent_data = recent_data.dropna(subset=['timestamp']).sort_values('timestamp')

                # Convert decimal to float
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in recent_data.columns:
                        recent_data[col] = recent_data[col].apply(
                            lambda x: float(x) if hasattr(x, '__float__') else x
                        )
            else:
                raise Exception(f"Unsupported database type: {db_config['type']}")

        # Add indicators
        from utils.indicators import add_technical_indicators
        recent_data = add_technical_indicators(recent_data)

        # Validate data length
        if len(recent_data) < self.sequence_length:
            raise Exception(f"Insufficient data for prediction. Need {self.sequence_length} records, got {len(recent_data)}")

        # Get latest sequence
        latest_data = recent_data.tail(self.sequence_length)[self.feature_columns].values
        scaled_data = self.scaler.transform(latest_data)

        # Reshape for LSTM input
        X_pred = scaled_data.reshape((1, self.sequence_length, len(self.feature_columns)))

        # Make prediction
        predicted_scaled = self.model.predict(X_pred, verbose=0)[0][0]

        # Inverse transform to get actual price
        predicted_close = self.target_scaler.inverse_transform([[predicted_scaled]])[0][0]

        # Calculate OHLC
        last_close = float(recent_data['close'].iloc[-1])
        change_pct = (predicted_close - last_close) / last_close

        predicted_open = last_close
        predicted_high = predicted_close * (1 + abs(change_pct) * 0.3)
        predicted_low = predicted_close * (1 - abs(change_pct) * 0.3)

        # Ensure high >= close >= low
        predicted_high = max(predicted_high, predicted_close)
        predicted_low = min(predicted_low, predicted_close)

        return {
            'open': float(predicted_open),
            'high': float(predicted_high),
            'low': float(predicted_low),
            'close': float(predicted_close),
            'volume': float(recent_data['volume'].tail(20).mean())
        }

# Singleton instance with lazy initialization
_predictor_instance = None

def predict_next_candle(symbol="BTCUSDT"):
    """Convenience function with singleton pattern (OPTIMIZED)"""
    global _predictor_instance
    
    if _predictor_instance is None:
        _predictor_instance = LSTMCryptoPredictor()
    
    _predictor_instance.symbol = symbol
    return _predictor_instance.predict_next_candle()
    
# import tensorflow as tf
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.optimizers import Adam
# import numpy as np
# import pandas as pd
# import os
# import requests
# import pickle
# import psycopg2
# from sklearn.preprocessing import MinMaxScaler
# from utils.indicators import add_technical_indicators
# from dotenv import load_dotenv

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

# class LSTMCryptoPredictor:
#     def __init__(self, symbol="BTCUSDT", sequence_length=60, model_path="models/saved_model.keras"):
#         self.symbol = symbol
#         self.sequence_length = sequence_length
#         self.model_path = model_path
#         self.scaler_path = model_path.replace('.keras', '_scaler.pkl')
#         self.target_scaler_path = model_path.replace('.keras', '_target_scaler.pkl')
#         self.model = None
#         self.scaler = MinMaxScaler(feature_range=(0, 1))  # For input features (22)
#         self.target_scaler = MinMaxScaler(feature_range=(0, 1))  # For target (1)
#         self.feature_columns = None

#         # Create models directory if it doesn't exist
#         os.makedirs(os.path.dirname(model_path), exist_ok=True)

#     def fetch_training_data(self, limit=1000):
#         """Fetch historical data from configured database for training"""
#         db_config = get_db_config()

#         if db_config["type"] == "mysql":
#             # Fetch from MySQL
#             connection = pymysql.connect(
#                 host=db_config["host"],
#                 user=db_config["user"],
#                 password=db_config["password"],
#                 database=db_config["database"],
#                 charset=db_config["charset"],
#                 cursorclass=pymysql.cursors.DictCursor
#             )
#             try:
#                 with connection.cursor() as cursor:
#                     cursor.execute(f"""
#                     SELECT timestamp, open, high, low, close, volume
#                     FROM crypto_candles
#                     WHERE symbol = '{self.symbol}'
#                     ORDER BY timestamp ASC
#                     LIMIT {limit}
#                     """)
#                     data = cursor.fetchall()
#             finally:
#                 connection.close()

#             if len(data) < self.sequence_length + 10:  # Need minimum data for training
#                 raise Exception(f"Insufficient data for training. Need at least {self.sequence_length + 10} records")

#             df = pd.DataFrame(data)
#             df['timestamp'] = pd.to_datetime(df['timestamp'])
#             df = df.sort_values('timestamp')

#         elif db_config["type"] == "postgresql":
#             # Fetch from PostgreSQL
#             try:
#                 if "connection_string" in db_config:
#                     connection = psycopg2.connect(db_config["connection_string"])
#                 else:
#                     connection = psycopg2.connect(
#                         host=db_config["host"],
#                         user=db_config["user"],
#                         password=db_config["password"],
#                         database=db_config["database"],
#                         port=db_config["port"]
#                     )
#                 with connection.cursor() as cursor:
#                     cursor.execute(f"""
#                     SELECT timestamp, open, high, low, close, volume
#                     FROM crypto_candles
#                     WHERE symbol = '{self.symbol}'
#                     ORDER BY timestamp ASC
#                     LIMIT {limit}
#                     """)
#                     data = cursor.fetchall()
#                     column_names = [desc[0] for desc in cursor.description]
#                     data = [dict(zip(column_names, row)) for row in data]
#             except psycopg2.Error as e:
#                 print(f"PostgreSQL query error: {e}")
#                 data = []
#             finally:
#                 if 'connection' in locals():
#                     connection.close()

#             if len(data) < self.sequence_length + 10:  # Need minimum data for training
#                 raise Exception(f"Insufficient data for training. Need at least {self.sequence_length + 10} records")

#             df = pd.DataFrame(data)
#             df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
#             df = df.sort_values('timestamp').dropna()

#         elif db_config["type"] == "questdb":
#             # Fetch from QuestDB
#             query = f"""
#             SELECT timestamp, open, high, low, close, volume
#             FROM crypto_candles
#             WHERE symbol = '{self.symbol}'
#             ORDER BY timestamp ASC
#             LIMIT {limit}
#             """
#             response = requests.get(f"{db_config['url']}/exec", params={'query': query})

#             if response.status_code != 200 or 'dataset' not in response.json():
#                 raise Exception("Failed to fetch training data from QuestDB")

#             data = response.json()['dataset']
#             if len(data) < self.sequence_length + 10:  # Need minimum data for training
#                 raise Exception(f"Insufficient data for training. Need at least {self.sequence_length + 10} records")

#             df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
#             df['timestamp'] = pd.to_datetime(df['timestamp'])
#             df = df.sort_values('timestamp')
#         else:
#             raise Exception(f"Unsupported database type: {db_config['type']}")

#         return df

#     def prepare_data(self, df):
#         """Prepare data with technical indicators and scaling"""
#         # Add technical indicators
#         df = add_technical_indicators(df)

#         # Define feature columns (OHLCV + technical indicators)
#         self.feature_columns = [
#             'open', 'high', 'low', 'close', 'volume',
#             'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi',
#             'macd', 'macd_signal', 'macd_hist',
#             'bb_upper', 'bb_middle', 'bb_lower', 'atr',
#             'stoch_k', 'stoch_d', 'willr', 'volume_sma', 'roc'
#         ]

#         # Ensure all feature columns exist
#         missing_cols = [col for col in self.feature_columns if col not in df.columns]
#         if missing_cols:
#             raise ValueError(f"Missing feature columns: {missing_cols}")

#         # Scale the features (22 features)
#         feature_data = df[self.feature_columns].values
#         scaled_data = self.scaler.fit_transform(feature_data)

#         # Scale the targets separately (1 feature)
#         targets = df['close'].shift(-1).values[:-1]  # Next close price
#         scaled_targets = self.target_scaler.fit_transform(targets.reshape(-1, 1))

#         return scaled_data[:-1], scaled_targets  # Remove last sample as we don't have target for it

#     def create_sequences(self, data, targets):
#         """Create sequences for LSTM input"""
#         X, y = [], []

#         for i in range(len(data) - self.sequence_length):
#             X.append(data[i:(i + self.sequence_length)])
#             y.append(targets[i + self.sequence_length])

#         return np.array(X), np.array(y)

#     def build_model(self, input_shape):
#         """Build LSTM model architecture"""
#         model = Sequential([
#             Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape)),
#             Dropout(0.2),
#             Bidirectional(LSTM(64, return_sequences=True)),
#             Dropout(0.2),
#             LSTM(32),
#             Dropout(0.2),
#             Dense(16, activation='relu'),
#             Dense(1)  # Predict next close price
#         ])

#         model.compile(optimizer=Adam(learning_rate=0.001),
#                      loss='mean_squared_error',
#                      metrics=['mae'])

#         return model

#     def train(self, epochs=100, batch_size=32, validation_split=0.2):
#         """Train the LSTM model"""
#         print(f"Fetching training data for {self.symbol}...")
#         df = self.fetch_training_data(limit=2000)  # Get more data for training

#         print(f"Preparing data with {len(df)} records...")
#         scaled_data, scaled_targets = self.prepare_data(df)

#         X, y = self.create_sequences(scaled_data, scaled_targets)

#         print(f"Training data shape: {X.shape}, Target shape: {y.shape}")

#         # Build model
#         self.model = self.build_model((X.shape[1], X.shape[2]))

#         # Callbacks
#         early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#         checkpoint = ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True)

#         # Train model
#         print("Training LSTM model...")
#         history = self.model.fit(
#             X, y,
#             epochs=epochs,
#             batch_size=batch_size,
#             validation_split=validation_split,
#             callbacks=[early_stopping, checkpoint],
#             verbose=1
#         )

#         # Save scalers
#         with open(self.scaler_path, 'wb') as f:
#             pickle.dump(self.scaler, f)
#         with open(self.target_scaler_path, 'wb') as f:
#             pickle.dump(self.target_scaler, f)

#         print(f"Model trained and saved to {self.model_path}")
#         print(f"Feature scaler saved to {self.scaler_path}")
#         print(f"Target scaler saved to {self.target_scaler_path}")
#         return history

#     def load_model(self):
#         """Load trained model"""
#         if os.path.exists(self.model_path):
#             self.model = load_model(self.model_path)
#             print(f"Model loaded from {self.model_path}")

#             # Load scalers
#             if os.path.exists(self.scaler_path) and os.path.exists(self.target_scaler_path):
#                 with open(self.scaler_path, 'rb') as f:
#                     self.scaler = pickle.load(f)
#                 with open(self.target_scaler_path, 'rb') as f:
#                     self.target_scaler = pickle.load(f)
#                 print(f"Scalers loaded from {self.scaler_path} and {self.target_scaler_path}")
#             else:
#                 raise Exception("Scalers not found. Please retrain the model.")

#             # Ensure feature_columns is set (needed for prediction)
#             if self.feature_columns is None:
#                 self.feature_columns = [
#                     'open', 'high', 'low', 'close', 'volume',
#                     'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi',
#                     'macd', 'macd_signal', 'macd_hist',
#                     'bb_upper', 'bb_middle', 'bb_lower', 'atr',
#                     'stoch_k', 'stoch_d', 'willr', 'volume_sma', 'roc'
#                 ]
#         else:
#             raise FileNotFoundError(f"Model file not found: {self.model_path}")

#     def predict_next_candle(self, recent_data=None):
#         """
#         Predict the next candle using recent data
#         recent_data: DataFrame with recent OHLCV data (if None, fetch from QuestDB)
#         """
#         if self.model is None:
#             try:
#                 self.load_model()
#             except FileNotFoundError:
#                 raise Exception("Model not trained yet. Please train the model first.")

#         # Get recent data
#         if recent_data is None:
#             db_config = get_db_config()

#             if db_config["type"] == "mysql":
#                 # Fetch recent data from MySQL
#                 connection = pymysql.connect(
#                     host=db_config["host"],
#                     user=db_config["user"],
#                     password=db_config["password"],
#                     database=db_config["database"],
#                     charset=db_config["charset"],
#                     cursorclass=pymysql.cursors.DictCursor
#                 )
#                 try:
#                     with connection.cursor() as cursor:
#                         cursor.execute(f"""
#                         SELECT timestamp, open, high, low, close, volume
#                         FROM crypto_candles
#                         WHERE symbol = '{self.symbol}'
#                         ORDER BY timestamp DESC
#                         LIMIT {self.sequence_length + 10}
#                         """)
#                         data = cursor.fetchall()
#                 finally:
#                     connection.close()

#                 recent_data = pd.DataFrame(data)
#                 recent_data['timestamp'] = pd.to_datetime(recent_data['timestamp'])
#                 # Convert decimal.Decimal to float for all numeric columns
#                 for col in ['open', 'high', 'low', 'close', 'volume']:
#                     if col in recent_data.columns:
#                         recent_data[col] = recent_data[col].astype(float)
#                 recent_data = recent_data.sort_values('timestamp')

#             elif db_config["type"] == "postgresql":
#                 # Fetch recent data from PostgreSQL
#                 try:
#                     if "connection_string" in db_config:
#                         connection = psycopg2.connect(db_config["connection_string"])
#                     else:
#                         connection = psycopg2.connect(
#                             host=db_config["host"],
#                             user=db_config["user"],
#                             password=db_config["password"],
#                             database=db_config["database"],
#                             port=db_config["port"]
#                         )
#                     with connection.cursor() as cursor:
#                         cursor.execute(f"""
#                         SELECT timestamp, open, high, low, close, volume
#                         FROM crypto_candles
#                         WHERE symbol = '{self.symbol}'
#                         ORDER BY timestamp DESC
#                         LIMIT {self.sequence_length + 10}
#                         """)
#                         data = cursor.fetchall()
#                         column_names = [desc[0] for desc in cursor.description]
#                         data = [dict(zip(column_names, row)) for row in data]
#                 except psycopg2.Error as e:
#                     print(f"PostgreSQL query error: {e}")
#                     data = []
#                 finally:
#                     if 'connection' in locals():
#                         connection.close()

#                 if not data:
#                     raise Exception(f"No data found for symbol {self.symbol} in PostgreSQL database")

#                 recent_data = pd.DataFrame(data)
#                 if 'timestamp' in recent_data.columns:
#                     recent_data['timestamp'] = pd.to_datetime(recent_data['timestamp'], errors='coerce')
#                     recent_data = recent_data.dropna(subset=['timestamp'])
#                     recent_data = recent_data.sort_values('timestamp')
#                 else:
#                     raise Exception("No timestamp column found in PostgreSQL data")

#                 # Convert decimal.Decimal to float for all numeric columns
#                 for col in ['open', 'high', 'low', 'close', 'volume']:
#                     if col in recent_data.columns:
#                         recent_data[col] = recent_data[col].apply(lambda x: float(x) if hasattr(x, '__float__') else x)

#             elif db_config["type"] == "questdb":
#                 # Fetch recent data from QuestDB
#                 query = f"""
#                 SELECT timestamp, open, high, low, close, volume
#                 FROM crypto_candles
#                 WHERE symbol = '{self.symbol}'
#                 ORDER BY timestamp DESC
#                 LIMIT {self.sequence_length + 10}
#                 """
#                 response = requests.get(f"{db_config['url']}/exec", params={'query': query})

#                 if response.status_code != 200:
#                     print(f"HTTP Error: {response.status_code}")
#                     print(f"Response: {response.text}")
#                     raise Exception("Failed to fetch recent data from QuestDB")

#                 try:
#                     json_response = response.json()
#                     if 'dataset' not in json_response:
#                         print(f"Unexpected response format: {json_response}")
#                         raise Exception("Failed to fetch recent data from QuestDB")
#                     data = json_response['dataset']
#                 except Exception as e:
#                     print(f"JSON parsing error: {e}")
#                     print(f"Raw response: {response.text}")
#                     raise Exception("Failed to fetch recent data from QuestDB")

#                 data = response.json()['dataset']
#                 recent_data = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
#                 recent_data['timestamp'] = pd.to_datetime(recent_data['timestamp'])
#                 recent_data = recent_data.sort_values('timestamp')
#             else:
#                 raise Exception(f"Unsupported database type: {db_config['type']}")

#         # Prepare data with indicators
#         recent_data = add_technical_indicators(recent_data)

#         # Get latest sequence
#         if len(recent_data) < self.sequence_length:
#             raise Exception(f"Insufficient data for prediction. Need at least {self.sequence_length} records")

#         latest_data = recent_data.tail(self.sequence_length)[self.feature_columns].values
#         scaled_data = self.scaler.transform(latest_data)

#         # Reshape for LSTM input
#         X_pred = np.array([scaled_data])
#         X_pred = X_pred.reshape((X_pred.shape[0], X_pred.shape[1], X_pred.shape[2]))

#         # Make prediction
#         predicted_scaled = self.model.predict(X_pred, verbose=0)[0][0]

#         # Inverse transform to get actual price
#         predicted_close = self.target_scaler.inverse_transform([[predicted_scaled]])[0][0]

#         # Get last known prices for OHLC calculation
#         last_close = float(recent_data['close'].iloc[-1])
#         last_high = float(recent_data['high'].iloc[-1])
#         last_low = float(recent_data['low'].iloc[-1])

#         # Estimate OHLC based on predicted close
#         volatility = recent_data['close'].pct_change().std()
#         change_pct = (predicted_close - last_close) / last_close

#         # Generate OHLC around the predicted close
#         predicted_open = last_close
#         predicted_high = predicted_close * (1 + abs(change_pct) * 0.3)
#         predicted_low = predicted_close * (1 - abs(change_pct) * 0.3)

#         # Ensure high >= close >= low
#         predicted_high = max(predicted_high, predicted_close)
#         predicted_low = min(predicted_low, predicted_close)

#         return {
#             'open': float(predicted_open),
#             'high': float(predicted_high),
#             'low': float(predicted_low),
#             'close': float(predicted_close),
#             'volume': float(recent_data['volume'].tail(20).mean())  # Average recent volume
#         }

# # Global predictor instance
# predictor = LSTMCryptoPredictor()

# def predict_next_candle(symbol="BTCUSDT"):
#     """Convenience function for predictions"""
#     predictor.symbol = symbol
#     return predictor.predict_next_candle()