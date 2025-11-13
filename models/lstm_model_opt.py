import numpy as np
import pandas as pd
import os
import requests
import pickle
import psycopg2
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# CRITICAL: Don't import TensorFlow at module level!
# Only import when actually needed to save memory

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
    else:
        return {
            "type": "questdb",
            "url": os.getenv("QUESTDB_URL", "http://localhost:9000")
        }

class LSTMCryptoPredictor:
    # Class-level cache for model and scalers (shared across instances)
    _model_cache = {}
    _scaler_cache = {}
    
    def __init__(self, symbol="BTCUSDT", sequence_length=30, model_path="models/saved_model.keras"):
        self.symbol = symbol
        self.sequence_length = sequence_length  # Reduced from 60 to 30
        self.model_path = model_path
        self.scaler_path = model_path.replace('.keras', '_scaler.pkl')
        self.target_scaler_path = model_path.replace('.keras', '_target_scaler.pkl')
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = None

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

    def fetch_training_data(self, limit=1000):
        """Fetch historical data with timeout"""
        db_config = get_db_config()

        if db_config["type"] == "postgresql":
            try:
                if "connection_string" in db_config:
                    connection = psycopg2.connect(
                        db_config["connection_string"],
                        connect_timeout=10  # Added timeout
                    )
                else:
                    connection = psycopg2.connect(
                        host=db_config["host"],
                        user=db_config["user"],
                        password=db_config["password"],
                        database=db_config["database"],
                        port=db_config["port"],
                        connect_timeout=10  # Added timeout
                    )
                    
                with connection.cursor() as cursor:
                    cursor.execute(f"""
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
                print(f"PostgreSQL query error: {e}")
                raise
            finally:
                if 'connection' in locals():
                    connection.close()

            if len(data) < self.sequence_length + 10:
                raise Exception(f"Insufficient data for training. Need at least {self.sequence_length + 10} records")

            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.sort_values('timestamp').dropna()

        else:
            raise Exception(f"Unsupported database type: {db_config['type']}")

        return df

    def prepare_data(self, df):
        """Prepare data with simplified technical indicators"""
        from utils.indicators import add_technical_indicators
        
        df = add_technical_indicators(df)

        # Reduced feature set for lower memory usage
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi',
            'macd', 'macd_signal', 'bb_upper', 'bb_lower'
        ]

        feature_data = df[self.feature_columns].values
        scaled_data = self.scaler.fit_transform(feature_data)

        targets = df['close'].shift(-1).values[:-1]
        scaled_targets = self.target_scaler.fit_transform(targets.reshape(-1, 1))

        return scaled_data[:-1], scaled_targets

    def create_sequences(self, data, targets):
        """Create sequences for LSTM input"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(targets[i + self.sequence_length])
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """Build OPTIMIZED lightweight LSTM model"""
        # Lazy import TensorFlow only when needed
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        
        # Much lighter architecture for Render
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=input_shape),  # Reduced from 128
            Dropout(0.2),
            LSTM(16),  # Reduced from 64
            Dropout(0.2),
            Dense(8, activation='relu'),  # Reduced from 16
            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )

        return model

    def train(self, epochs=50, batch_size=16, validation_split=0.2):
        """Train the LSTM model with reduced epochs"""
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        
        print(f"Fetching training data for {self.symbol}...")
        df = self.fetch_training_data(limit=1000)  # Reduced from 2000

        print(f"Preparing data with {len(df)} records...")
        scaled_data, scaled_targets = self.prepare_data(df)

        X, y = self.create_sequences(scaled_data, scaled_targets)
        print(f"Training data shape: {X.shape}, Target shape: {y.shape}")

        self.model = self.build_model((X.shape[1], X.shape[2]))

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,  # Reduced from 10
            restore_best_weights=True
        )
        checkpoint = ModelCheckpoint(
            self.model_path,
            monitor='val_loss',
            save_best_only=True
        )

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

        print(f"Model saved to {self.model_path}")
        
        # Clear TensorFlow session to free memory
        tf.keras.backend.clear_session()
        
        return history

    def load_model(self):
        """Load model with caching"""
        cache_key = self.model_path
        
        # Check cache first
        if cache_key in self._model_cache:
            print(f"Loading model from cache: {cache_key}")
            self.model = self._model_cache[cache_key]['model']
            self.scaler = self._model_cache[cache_key]['scaler']
            self.target_scaler = self._model_cache[cache_key]['target_scaler']
            self.feature_columns = self._model_cache[cache_key]['features']
            return

        if os.path.exists(self.model_path):
            # Lazy import TensorFlow
            from tensorflow.keras.models import load_model
            
            print(f"Loading model from disk: {self.model_path}")
            self.model = load_model(self.model_path)

            if os.path.exists(self.scaler_path) and os.path.exists(self.target_scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                with open(self.target_scaler_path, 'rb') as f:
                    self.target_scaler = pickle.load(f)
            else:
                raise Exception("Scalers not found. Please retrain the model.")

            self.feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi',
                'macd', 'macd_signal', 'bb_upper', 'bb_lower'
            ]
            
            # Cache the loaded model
            self._model_cache[cache_key] = {
                'model': self.model,
                'scaler': self.scaler,
                'target_scaler': self.target_scaler,
                'features': self.feature_columns
            }
            
            print(f"Model cached successfully")
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

    def predict_next_candle(self, recent_data=None):
        """Predict next candle with memory optimization"""
        if self.model is None:
            try:
                self.load_model()
            except FileNotFoundError:
                raise Exception("Model not trained yet. Please train the model first.")

        if recent_data is None:
            db_config = get_db_config()

            if db_config["type"] == "postgresql":
                try:
                    if "connection_string" in db_config:
                        connection = psycopg2.connect(
                            db_config["connection_string"],
                            connect_timeout=10  # Added timeout
                        )
                    else:
                        connection = psycopg2.connect(
                            host=db_config["host"],
                            user=db_config["user"],
                            password=db_config["password"],
                            database=db_config["database"],
                            port=db_config["port"],
                            connect_timeout=10  # Added timeout
                        )
                        
                    with connection.cursor() as cursor:
                        cursor.execute(f"""
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
                    print(f"PostgreSQL query error: {e}")
                    raise
                finally:
                    if 'connection' in locals():
                        connection.close()

                if not data:
                    raise Exception(f"No data found for symbol {self.symbol}")

                recent_data = pd.DataFrame(data)
                recent_data['timestamp'] = pd.to_datetime(recent_data['timestamp'], errors='coerce')
                recent_data = recent_data.dropna(subset=['timestamp']).sort_values('timestamp')

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

        if len(recent_data) < self.sequence_length:
            raise Exception(f"Insufficient data. Need {self.sequence_length} records")

        latest_data = recent_data.tail(self.sequence_length)[self.feature_columns].values
        scaled_data = self.scaler.transform(latest_data)

        X_pred = np.array([scaled_data]).reshape((1, self.sequence_length, len(self.feature_columns)))

        # Make prediction
        predicted_scaled = self.model.predict(X_pred, verbose=0)[0][0]
        predicted_close = self.target_scaler.inverse_transform([[predicted_scaled]])[0][0]

        # Calculate OHLC
        last_close = float(recent_data['close'].iloc[-1])
        change_pct = (predicted_close - last_close) / last_close

        predicted_open = last_close
        predicted_high = predicted_close * (1 + abs(change_pct) * 0.3)
        predicted_low = predicted_close * (1 - abs(change_pct) * 0.3)

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
    """Convenience function with singleton pattern"""
    global _predictor_instance
    
    if _predictor_instance is None:
        _predictor_instance = LSTMCryptoPredictor()
    
    _predictor_instance.symbol = symbol
    return _predictor_instance.predict_next_candle()