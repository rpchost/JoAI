import talib
import numpy as np
import pandas as pd

def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe for LSTM features
    """
    # Ensure we have OHLC data
    if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        raise ValueError("DataFrame must contain OHLC columns")

    # Convert to numpy arrays for TA-Lib
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    open_price = df['open'].values.astype(float)
    volume = df['volume'].values.astype(float) if 'volume' in df.columns else np.ones(len(df))

    # Moving Averages
    df['sma_20'] = talib.SMA(close, timeperiod=20)
    df['sma_50'] = talib.SMA(close, timeperiod=50)
    df['ema_12'] = talib.EMA(close, timeperiod=12)
    df['ema_26'] = talib.EMA(close, timeperiod=26)

    # RSI (Relative Strength Index)
    df['rsi'] = talib.RSI(close, timeperiod=14)

    # MACD (Moving Average Convergence Divergence)
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macd_signal'] = macdsignal
    df['macd_hist'] = macdhist

    # Bollinger Bands
    upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['bb_upper'] = upperband
    df['bb_middle'] = middleband
    df['bb_lower'] = lowerband

    # ATR (Average True Range) - Volatility
    df['atr'] = talib.ATR(high, low, close, timeperiod=14)

    # Stochastic Oscillator
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
    df['stoch_k'] = slowk
    df['stoch_d'] = slowd

    # Williams %R
    df['willr'] = talib.WILLR(high, low, close, timeperiod=14)

    # Volume indicators
    df['volume_sma'] = talib.SMA(volume, timeperiod=20)

    # Price momentum
    df['roc'] = talib.ROC(close, timeperiod=10)  # Rate of change

    # Fill NaN values with forward fill, then backward fill for remaining NaNs
    df = df.ffill().bfill()

    return df