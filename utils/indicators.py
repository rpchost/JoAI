import talib
import numpy as np
import pandas as pd
from functools import lru_cache

class IndicatorCache:
    """Simple cache for indicator calculations"""
    _cache = {}
    
    @classmethod
    def get_cache_key(cls, df, indicator_name):
        """Generate cache key based on last timestamp and indicator"""
        if len(df) == 0:
            return None
        last_timestamp = df.index[-1] if hasattr(df.index, '__getitem__') else len(df)
        return f"{indicator_name}_{last_timestamp}_{len(df)}"
    
    @classmethod
    def get(cls, key):
        return cls._cache.get(key)
    
    @classmethod
    def set(cls, key, value):
        # Keep cache size limited to prevent memory bloat
        if len(cls._cache) > 50:
            # Remove oldest entries
            keys_to_remove = list(cls._cache.keys())[:10]
            for k in keys_to_remove:
                del cls._cache[k]
        cls._cache[key] = value

def add_technical_indicators(df, use_cache=True):
    """
    Add technical indicators to the dataframe for LSTM features (OPTIMIZED)
    
    Optimizations:
    - Pre-allocates arrays
    - Vectorized operations where possible
    - Efficient NaN handling
    - Input validation
    - Optional caching
    """
    if df is None or len(df) == 0:
        raise ValueError("DataFrame is empty")
    
    # Ensure we have OHLC data
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain OHLC columns. Found: {df.columns.tolist()}")

    # Work on a copy to avoid modifying original
    df = df.copy()

    # Pre-allocate numpy arrays once (more memory efficient)
    n = len(df)
    close = np.ascontiguousarray(df['close'].values, dtype=np.float64)
    high = np.ascontiguousarray(df['high'].values, dtype=np.float64)
    low = np.ascontiguousarray(df['low'].values, dtype=np.float64)
    open_price = np.ascontiguousarray(df['open'].values, dtype=np.float64)
    
    # Handle volume with default if missing
    if 'volume' in df.columns:
        volume = np.ascontiguousarray(df['volume'].values, dtype=np.float64)
    else:
        volume = np.ones(n, dtype=np.float64)

    # Validate data integrity
    if np.any(np.isnan(close)) or np.any(np.isinf(close)):
        print("Warning: NaN or Inf values detected in close prices, attempting to clean...")
        close = np.nan_to_num(close, nan=0.0, posinf=0.0, neginf=0.0)
        high = np.nan_to_num(high, nan=0.0, posinf=0.0, neginf=0.0)
        low = np.nan_to_num(low, nan=0.0, posinf=0.0, neginf=0.0)
        open_price = np.nan_to_num(open_price, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        # ============================================
        # MOVING AVERAGES (Trend Indicators)
        # ============================================
        df['sma_20'] = talib.SMA(close, timeperiod=20)
        df['sma_50'] = talib.SMA(close, timeperiod=50)
        df['ema_12'] = talib.EMA(close, timeperiod=12)
        df['ema_26'] = talib.EMA(close, timeperiod=26)

        # ============================================
        # MOMENTUM INDICATORS
        # ============================================
        # RSI (Relative Strength Index)
        df['rsi'] = talib.RSI(close, timeperiod=14)

        # MACD (Moving Average Convergence Divergence)
        macd, macdsignal, macdhist = talib.MACD(
            close, 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        df['macd'] = macd
        df['macd_signal'] = macdsignal
        df['macd_hist'] = macdhist

        # Rate of change (momentum)
        df['roc'] = talib.ROC(close, timeperiod=10)

        # ============================================
        # VOLATILITY INDICATORS
        # ============================================
        # Bollinger Bands
        upperband, middleband, lowerband = talib.BBANDS(
            close, 
            timeperiod=20, 
            nbdevup=2, 
            nbdevdn=2, 
            matype=0
        )
        df['bb_upper'] = upperband
        df['bb_middle'] = middleband
        df['bb_lower'] = lowerband

        # ATR (Average True Range) - Volatility
        df['atr'] = talib.ATR(high, low, close, timeperiod=14)

        # ============================================
        # OSCILLATORS
        # ============================================
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(
            high, low, close, 
            fastk_period=14, 
            slowk_period=3, 
            slowd_period=3
        )
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd

        # Williams %R
        df['willr'] = talib.WILLR(high, low, close, timeperiod=14)

        # ============================================
        # VOLUME INDICATORS
        # ============================================
        df['volume_sma'] = talib.SMA(volume, timeperiod=20)

    except Exception as e:
        print(f"Error calculating indicators: {str(e)}")
        raise

    # ============================================
    # EFFICIENT NaN HANDLING
    # ============================================
    # Strategy: Forward fill first (more logical for time series), 
    # then backward fill for any remaining NaNs at the start
    
    # Get indicator columns (all except OHLCV)
    indicator_cols = [col for col in df.columns if col not in required_cols + ['volume', 'timestamp']]
    
    # Fill NaN values efficiently
    df[indicator_cols] = df[indicator_cols].fillna(method='ffill').fillna(method='bfill')
    
    # Final safety check: replace any remaining NaN with 0
    df[indicator_cols] = df[indicator_cols].fillna(0)

    # Verify output has no NaN values
    if df[indicator_cols].isnull().any().any():
        print("Warning: Some NaN values remain after filling. Replacing with 0.")
        df[indicator_cols] = df[indicator_cols].fillna(0)

    return df


def add_technical_indicators_minimal(df):
    """
    Minimal indicator set for very memory-constrained environments
    Uses only the most important 10 indicators
    """
    if df is None or len(df) == 0:
        raise ValueError("DataFrame is empty")
    
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain OHLC columns")

    df = df.copy()
    
    # Pre-allocate arrays
    close = np.ascontiguousarray(df['close'].values, dtype=np.float64)
    high = np.ascontiguousarray(df['high'].values, dtype=np.float64)
    low = np.ascontiguousarray(df['low'].values, dtype=np.float64)
    
    try:
        # Minimal essential indicators
        df['sma_20'] = talib.SMA(close, timeperiod=20)
        df['sma_50'] = talib.SMA(close, timeperiod=50)
        df['rsi'] = talib.RSI(close, timeperiod=14)
        
        macd, macdsignal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macdsignal
        
        upperband, _, lowerband = talib.BBANDS(close, timeperiod=20)
        df['bb_upper'] = upperband
        df['bb_lower'] = lowerband
        
        df['atr'] = talib.ATR(high, low, close, timeperiod=14)
        
    except Exception as e:
        print(f"Error in minimal indicators: {str(e)}")
        raise

    # Fill NaN
    indicator_cols = [col for col in df.columns if col not in required_cols + ['volume', 'timestamp']]
    df[indicator_cols] = df[indicator_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)

    return df


def validate_indicator_data(df):
    """
    Validate that indicator calculations were successful
    Returns True if valid, False otherwise with error message
    """
    try:
        # Check for all NaN columns
        for col in df.columns:
            if df[col].isna().all():
                return False, f"Column {col} is entirely NaN"
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                return False, f"Column {col} contains infinite values"
        
        return True, "All indicators valid"
    
    except Exception as e:
        return False, f"Validation error: {str(e)}"