import os
import re
import random
from typing import Dict, Optional, List
from datetime import datetime
from models.lstm_model import predict_next_candle
import psycopg2
import pandas as pd
from models.lstm_model import FEATURE_COLUMNS

# === TOXICITY DETECTION & ESCALATION SYSTEM ===
TOXIC_WORDS = [
    # English
    "fuck", "shit", "bitch", "cunt", "asshole", "motherfucker", "nigger", "faggot", "retard",
    "pussy", "dick", "cock", "whore", "slut", "bastard", "crap", "go to hell", "gotohell",
    # Variations & common bypasses
    "fuk", "fck", "fukin", "f*ck", "sh1t", "b1tch", "a$$", "nigg", "f@g", "r3tard",
    # Aggressive/offensive intent
    "kill yourself", "kys", "die", "hope you die", "stupid", "idiot", "dumb", "braindead",
    "scam", "scammer", "fraud", "fake", "garbage", "trash", "dogshit", "worthless"
]

  # 1. DYNAMIC COIN LIST (add this near the top)
COINS_READABLE = {
        'BTCUSDT': 'Bitcoin (BTC)', 'ETHUSDT': 'Ethereum (ETH)', 'BNBUSDT': 'BNB',
        'ADAUSDT': 'Cardano (ADA)', 'SOLUSDT': 'Solana (SOL)', 'XRPUSDT': 'XRP',
        'DOGEUSDT': 'Dogecoin (DOGE)', 'SHIBUSDT': 'Shiba Inu (SHIB)',
        'PEPEUSDT': 'Pepe (PEPE)', 'LINKUSDT': 'Chainlink (LINK)',
        'AVAXUSDT': 'Avalanche (AVAX)', 'TONUSDT': 'Toncoin (TON)'
    }

ALL_COINS_STR = " • ".join([name.split('(')[0].strip() for name in COINS_READABLE.values()])
ALL_COINS_EMOJI = "BTC • ETH • SOL • XRP • DOGE • SHIB • PEPE • LINK • AVAX • TON • ADA • BNB"

class JoAIConversationNLP:
    """
    Human-like conversational AI for crypto predictions
    Handles greetings, small talk, context, predictions, and technical analysis
    """
      
    def __init__(self, api_base_url: str = "http://localhost:8081"):
        self.api_base_url = api_base_url.rstrip('/')
        self.conversation_context = {
            'last_query': None,
            'last_symbol': None,
            'last_prediction': None,
            'user_name': None,
            'conversation_started': datetime.now()
        }
        
        # Symbol mappings
        self.symbol_map = {
            'btc': 'BTCUSDT', 'bitcoin': 'BTCUSDT', 'btcusdt': 'BTCUSDT',
            'eth': 'ETHUSDT', 'ethereum': 'ETHUSDT', 'ethusdt': 'ETHUSDT',
            'bnb': 'BNBUSDT', 'binance': 'BNBUSDT', 'bnbusdt': 'BNBUSDT',
            'ada': 'ADAUSDT', 'cardano': 'ADAUSDT', 'adausdt': 'ADAUSDT',
            'sol': 'SOLUSDT', 'solana': 'SOLUSDT', 'solusdt': 'SOLUSDT',
            'xrp': 'XRPUSDT', 'ripple': 'XRPUSDT',
            'doge': 'DOGEUSDT', 'dogecoin': 'DOGEUSDT', 'dogeusdt': 'DOGEUSDT',
            'shib': 'SHIBUSDT', 'shiba': 'SHIBUSDT', 'shibainu': 'SHIBUSDT',
            'pepe': 'PEPEUSDT', 'pepeusdt': 'PEPEUSDT',
            'link': 'LINKUSDT', 'chainlink': 'LINKUSDT',
            'avax': 'AVAXUSDT', 'avalanche': 'AVAXUSDT',
            'ton': 'TONUSDT', 'toncoin': 'TONUSDT'
        }
        
        # Timeframe patterns
        self.timeframe_map = {
            '1m': '1 minute', '5m': '5 minutes', '15m': '15 minutes', '30m': '30 minutes',
            '1h': '1 hour', '4h': '4 hours', '1d': '1 day',
            'minute': '1 minute', 'minutes': '5 minutes',
            'hour': '1 hour', 'hours': '4 hours',
            'day': '1 day', 'daily': '1 day',
        }
        
        # Technical Indicators Information
        self.technical_indicators = {
            'trend': {
                'SMA 20': {
                    'name': 'Simple Moving Average (20 periods)',
                    'description': 'Average price over the last 20 periods. Helps identify trend direction.',
                    'category': 'Trend Following'
                },
                'SMA 50': {
                    'name': 'Simple Moving Average (50 periods)',
                    'description': 'Average price over the last 50 periods. Used for longer-term trend analysis.',
                    'category': 'Trend Following'
                },
                'EMA 12': {
                    'name': 'Exponential Moving Average (12 periods)',
                    'description': 'Weighted average giving more importance to recent prices.',
                    'category': 'Trend Following'
                },
                'EMA 26': {
                    'name': 'Exponential Moving Average (26 periods)',
                    'description': 'Slower EMA used in MACD calculation.',
                    'category': 'Trend Following'
                }
            },
            'momentum': {
                'RSI': {
                    'name': 'Relative Strength Index',
                    'description': 'Measures momentum from 0-100. Above 70 = overbought, below 30 = oversold.',
                    'category': 'Momentum'
                },
                'MACD': {
                    'name': 'Moving Average Convergence Divergence',
                    'description': 'Shows relationship between two moving averages. Indicates trend changes.',
                    'category': 'Momentum'
                },
                'MACD Signal': {
                    'name': 'MACD Signal Line',
                    'description': '9-period EMA of MACD. Crossovers generate buy/sell signals.',
                    'category': 'Momentum'
                },
                'MACD Histogram': {
                    'name': 'MACD Histogram',
                    'description': 'Difference between MACD and signal line. Shows momentum strength.',
                    'category': 'Momentum'
                },
                'ROC': {
                    'name': 'Rate of Change',
                    'description': 'Measures percentage price change over time. Shows momentum speed.',
                    'category': 'Momentum'
                }
            },
            'volatility': {
                'BB Upper': {
                    'name': 'Bollinger Band Upper',
                    'description': 'Upper boundary of price volatility (2 std dev above SMA).',
                    'category': 'Volatility'
                },
                'BB Middle': {
                    'name': 'Bollinger Band Middle',
                    'description': 'Middle line of Bollinger Bands (20-period SMA).',
                    'category': 'Volatility'
                },
                'BB Lower': {
                    'name': 'Bollinger Band Lower',
                    'description': 'Lower boundary of price volatility (2 std dev below SMA).',
                    'category': 'Volatility'
                },
                'ATR': {
                    'name': 'Average True Range',
                    'description': 'Measures market volatility. Higher ATR = more volatile market.',
                    'category': 'Volatility'
                }
            },
            'oscillators': {
                'Stochastic K': {
                    'name': 'Stochastic Oscillator %K',
                    'description': 'Compares closing price to price range. Values 0-100.',
                    'category': 'Oscillator'
                },
                'Stochastic D': {
                    'name': 'Stochastic Oscillator %D',
                    'description': '3-period moving average of %K. Smoother signal line.',
                    'category': 'Oscillator'
                },
                'Williams %R': {
                    'name': 'Williams %R',
                    'description': 'Momentum indicator showing overbought/oversold levels (-100 to 0).',
                    'category': 'Oscillator'
                }
            },
            'volume': {
                'Volume': {
                    'name': 'Trading Volume',
                    'description': 'Number of shares/coins traded. Confirms trend strength.',
                    'category': 'Volume'
                },
                'Volume SMA': {
                    'name': 'Volume Simple Moving Average',
                    'description': 'Average trading volume over 20 periods. Identifies unusual activity.',
                    'category': 'Volume'
                }
            },
            'base': {
                'Open': {
                    'name': 'Opening Price',
                    'description': 'First traded price in the period.',
                    'category': 'Price Data'
                },
                'High': {
                    'name': 'Highest Price',
                    'description': 'Highest price reached during the period.',
                    'category': 'Price Data'
                },
                'Low': {
                    'name': 'Lowest Price',
                    'description': 'Lowest price reached during the period.',
                    'category': 'Price Data'
                },
                'Close': {
                    'name': 'Closing Price',
                    'description': 'Last traded price in the period. Most important for analysis.',
                    'category': 'Price Data'
                }
            }
        }
        
        # Greeting patterns
        self.greetings = {
            'patterns': [r'\b(hi|hello|hey|greetings|good\s+(morning|afternoon|evening|day)|howdy|yo|sup|what\'s up)\b'],
            'responses': [
                f"Hey! I'm JoAI — your crypto prediction AI with 12 fully trained models ready.\n{ALL_COINS_EMOJI}\nWhat are we predicting today?",
                f"Yo! Just finished training on 2000+ candles for each coin.\n{ALL_COINS_EMOJI}\nName your fighter.",
                f"Sup! I now predict {len(COINS_READABLE)} coins with deep LSTM brains. DOGE? PEPE? BTC? You name it — I got it.",
                f"Hey there! Welcome to the new JoAI — now covering:\n{ALL_COINS_EMOJI}\nWhich one should we analyze first?",
                f"Hello! I can predict the next move for BTC, ETH, SOL, XRP, DOGE, SHIB, PEPE and 5 more. What’s cooking?"
            ]
        }
        
        # Gratitude patterns
        self.gratitude = {
            'patterns': [
                r'\b(thank|thanks|thank you|thx|tysm|appreciate|grateful)\b',
            ],
            'responses': [
                "You're welcome! 😊 Anything else you'd like to know?",
                "Happy to help! Feel free to ask about any crypto predictions!",
                "My pleasure! Let me know if you need more predictions! 🚀",
                "Glad I could help! What else can I predict for you?",
            ]
        }
        
        # How are you patterns
        self.wellbeing = {
            'patterns': [
                r'\b(how are you|how\'re you|how r u|hows it going|what\'s up|wassup|sup)\b',
            ],
            'responses': [
                "I'm doing great, thanks for asking! 😊 Ready to help you with crypto predictions. What would you like to know?",
                "I'm fantastic! The crypto markets never sleep and neither do I! 🚀 How can I assist you?",
                "All systems operational! 💪 Let's predict some crypto prices together!",
                "Excellent! Excited to help you with predictions. Which cryptocurrency interests you?",
            ]
        }
        
        # Capability questions
        self.capabilities = {
            'patterns': [r'\b(what can you do|what do you do|help|capabilities|features|how does this work|how to use)\b'],
            'responses': [f"""Hey! I'm JoAI — the most advanced open-source crypto prediction AI in 2025.

        I predict the next candle for **{len(COINS_READABLE)} coins** using bidirectional LSTM + 22 indicators:

        {ALL_COINS_EMOJI}

        Just say:
        • "predict doge"
        • "what will pepe do?"
        • "btc next hour"
        • "shib pump or dump?"

        I understand natural language, remember context, and give you confidence levels.

        Try me — ask about any coin above!"""
            ]
        }

        # Technical Indicator queries
        self.technical_queries = {
            'patterns': [
                r'\b(indicator|indicators|technical|analysis|feature|features|input|inputs)\b',
                r'\b(what.*use|using|used)\b.*\b(predict|prediction|model|analysis)\b',
                r'\b(show|list|tell).*\b(indicator|technical|feature)\b',
                r'\b(my|your|the)\s*(indicator|technical|feature)\b',
            ]
        }
        
        # Model accuracy/performance queries
        self.model_info = {
            'patterns': [
                r'\b(accurate|accuracy|reliable|confidence|trust|performance|good|work)\b',
                r'\b(how (does|do) (it|this|you|the model) work)\b',
                r'\b(lstm|neural network|machine learning|ai|model)\b',
            ]
        }
        
        # Specific indicator explanation patterns
        self.indicator_explain = {
            'patterns': [
                r'\b(what is|explain|describe|tell me about)\s+(rsi|macd|sma|ema|bollinger|atr|stochastic|williams)\b',
            ]
        }
        
        # Goodbye patterns
        self.farewell = {
            'patterns': [
                r'\b(bye|goodbye|see you|see ya|later|gotta go|gtg|cya|farewell)\b',
            ],
            'responses': [
                "Goodbye! 👋 Come back anytime for crypto insights!",
                "See you later! Happy trading! 📈",
                "Take care! May your predictions be accurate! 🚀",
                "Bye! Feel free to return whenever you need predictions!",
            ]
        }
        
        # Jokes/personality
        self.jokes = {
            'patterns': [
                r'\b(joke|funny|laugh|humor|make me laugh)\b',
            ],
            'responses': [
                "Why did Bitcoin go to therapy? It had too many issues with its blocks! 😄 But seriously, what would you like to predict?",
                "What do you call a cryptocurrency that's always cold? Chill-coin! ❄️ Now, how can I help with predictions?",
                "I'd tell you a joke about Ethereum, but you might not get the gas fees! ⛽ What crypto interests you?",
            ]
        }
        
        # Confusion/unclear
        self.confusion_responses = [
            f"Hmm, not sure what you mean. Try asking about one of these coins:\n{ALL_COINS_EMOJI}",
            f"I didn't catch that. Want a prediction? I can do BTC, ETH, SOL, XRP, DOGE, SHIB, PEPE, LINK, AVAX, TON, ADA, BNB",
            f"Not sure! But I can predict the next move for any of these:\n{ALL_COINS_EMOJI}\nJust name a coin!",
            f"Lost me there! But I’m really good at predicting {ALL_COINS_STR}. Which one?",
        ]

        # === TOXICITY TRACKER (per session) ===
        self.toxicity_level = 0  # 0 = clean, 1 = warned, 2 = final warning, 3 = muted
        self.has_been_warned = False

    def is_toxic(self, query: str) -> bool:
        """Detect profanity or aggressive language"""
        query_lower = query.lower()
        return any(word in query_lower for word in TOXIC_WORDS)

    def handle_toxicity(self, query: str) -> Optional[Dict]:
        """Return a warning/response if message is toxic"""
        if not self.is_toxic(query):
            return None

        self.toxicity_level += 1

        if self.toxicity_level == 1:
            return {
                'success': True,
                'intent': 'toxicity_warning',
                'message': "Hey there! I noticed some strong language. Let's keep this chat respectful and positive — we're all here to talk crypto and have a good time! No warnings next time, just good vibes only!"
            }
        elif self.toxicity_level == 2:
            return {
                'success': True,
                'intent': 'toxicity_final_warning',
                'message': "Warning: This is your final warning. Continued use of inappropriate, offensive, or toxic language will result in your messages being ignored and may lead to a report and account suspension on the platform. Let's keep it clean and professional — I’m here to help with predictions!"
            }
        else:
            return {
                'success': True,
                'intent': 'toxicity_muted',
                'message': "Due to repeated inappropriate behavior, I can no longer respond to your messages. Please respect the community guidelines."
            }
        
    def detect_intent(self, query: str) -> str:
        """Classify user intent"""
        query_lower = query.lower().strip()
        
        # Check for greetings
        for pattern in self.greetings['patterns']:
            if re.search(pattern, query_lower):
                return 'greeting'
        
        # Check for gratitude
        for pattern in self.gratitude['patterns']:
            if re.search(pattern, query_lower):
                return 'gratitude'
        
        # Check for wellbeing
        for pattern in self.wellbeing['patterns']:
            if re.search(pattern, query_lower):
                return 'wellbeing'
        
        # Check for capabilities
        for pattern in self.capabilities['patterns']:
            if re.search(pattern, query_lower):
                return 'capabilities'
        
        # Check for farewell
        for pattern in self.farewell['patterns']:
            if re.search(pattern, query_lower):
                return 'farewell'
        
        # Check for jokes
        for pattern in self.jokes['patterns']:
            if re.search(pattern, query_lower):
                return 'joke'
        
        # Check for specific indicator explanation
        for pattern in self.indicator_explain['patterns']:
            if re.search(pattern, query_lower):
                return 'indicator_explain'
        
        # Check for technical indicator queries
        for pattern in self.technical_queries['patterns']:
            if re.search(pattern, query_lower):
                return 'technical_indicators'
        
        # Check for model info
        for pattern in self.model_info['patterns']:
            if re.search(pattern, query_lower):
                return 'model_info'
        
        # Check for prediction request
        prediction_keywords = ['predict', 'prediction', 'forecast', 'what will', 'price', 'value', 'worth', 'next']
        if any(keyword in query_lower for keyword in prediction_keywords):
            return 'prediction'
        
        # Check if just a symbol mentioned
        if self.extract_symbol(query):
            return 'prediction'
        
        return 'unknown'

    def get_technical_indicators_response(self) -> str:
        """Generate comprehensive technical indicators response"""
        response = """🔬 **Technical Indicators Used in My LSTM Model**

I analyze **22 different features** to make accurate predictions:

📈 **Trend Indicators** (4):
   • **SMA 20** - Simple Moving Average (20 periods)
   • **SMA 50** - Simple Moving Average (50 periods)  
   • **EMA 12** - Fast Exponential Moving Average
   • **EMA 26** - Slow Exponential Moving Average

⚡ **Momentum Indicators** (5):
   • **RSI** - Relative Strength Index (overbought/oversold)
   • **MACD** - Moving Average Convergence Divergence
   • **MACD Signal** - MACD signal line
   • **MACD Histogram** - MACD momentum strength
   • **ROC** - Rate of Change (momentum speed)

📊 **Volatility Indicators** (4):
   • **Bollinger Upper** - Upper volatility band
   • **Bollinger Middle** - Middle band (20 SMA)
   • **Bollinger Lower** - Lower volatility band
   • **ATR** - Average True Range (market volatility)

🎯 **Oscillators** (3):
   • **Stochastic %K** - Fast stochastic oscillator
   • **Stochastic %D** - Slow stochastic oscillator
   • **Williams %R** - Williams Percent Range

📦 **Volume Indicators** (2):
   • **Volume** - Trading volume
   • **Volume SMA** - Average volume (20 periods)

💰 **Price Data** (4):
   • **Open** - Opening price
   • **High** - Highest price
   • **Low** - Lowest price
   • **Close** - Closing price

🧠 **My Neural Network**: Bidirectional LSTM with 128+64+32 units
📚 **Training**: 2000+ historical data points per symbol
⚙️ **Technology**: TensorFlow + TA-Lib + Scikit-learn

Want to know more about a specific indicator? Just ask! 🚀"""
        
        return response

    def get_model_info_response(self) -> str:
        """Generate model information response"""
        response = """🤖 **About My Prediction Model**

**Architecture**: Advanced LSTM Neural Network
   • **Bidirectional LSTM** layers (128 → 64 → 32 units)
   • **Dropout layers** (0.2) to prevent overfitting
   • **Dense layers** for final prediction

**Training Process**:
   • 📚 Trained on 2000+ historical candles per cryptocurrency
   • 🎯 Uses 22 technical indicators + price data
   • ⚡ Early stopping to find optimal performance
   • 💾 Validates on 20% holdout data

**How It Works**:
   1. Analyzes last 60 candles of price history
   2. Calculates 22 technical indicators
   3. Normalizes data with MinMax scaling
   4. LSTM processes sequential patterns
   5. Predicts next candle (Open, High, Low, Close, Volume)

**Accuracy**:
   • Optimized for short-term predictions (1m - 1h)
   • Best for trending markets
   • Combines multiple indicator signals
   • Continuously learning from new data

**Technologies**:
   • 🧠 TensorFlow/Keras - Deep Learning
   • 📊 TA-Lib - Technical Analysis
   • 🔢 Scikit-learn - Data Preprocessing
   • 💾 PostgreSQL - Data Storage

**Limitations**:
   ⚠️ Past performance doesn't guarantee future results
   ⚠️ Market news can cause unexpected movements
   ⚠️ Use predictions as one tool among many

Want to see the technical indicators I use? Just ask! 📈"""
        
        return response

    def explain_indicator(self, query: str) -> Optional[str]:
        """Explain a specific technical indicator"""
        query_lower = query.lower()
        
        # Map query terms to indicator keys
        indicator_mapping = {
            'rsi': ('momentum', 'RSI'),
            'macd': ('momentum', 'MACD'),
            'sma': ('trend', 'SMA 20'),
            'ema': ('trend', 'EMA 12'),
            'bollinger': ('volatility', 'BB Upper'),
            'atr': ('volatility', 'ATR'),
            'stochastic': ('oscillators', 'Stochastic K'),
            'williams': ('oscillators', 'Williams %R'),
        }
        
        for keyword, (category, indicator_key) in indicator_mapping.items():
            if keyword in query_lower:
                indicator = self.technical_indicators[category][indicator_key]
                
                # Get related indicators
                related = [k for k in self.technical_indicators[category].keys() if k != indicator_key]
                
                response = f"""📊 **{indicator['name']}**

**Description**: {indicator['description']}

**Category**: {indicator['category']}

**How I Use It**: I include this indicator as one of my 22 input features. My LSTM neural network analyzes patterns in this indicator over the last 60 time periods to predict future price movements.

**Related Indicators**: {', '.join(related[:3]) if related else 'None'}

Want to know more? Ask me about any other indicator! 🚀"""
                
                return response
        
        return None

    def extract_symbol(self, text: str) -> Optional[str]:
        """Extract crypto symbol from text"""
        text_lower = text.lower().strip()
        
        # Direct match
        if text_lower in self.symbol_map:
            return self.symbol_map[text_lower]
        
        # Already formatted as USDT pair
        if text_lower.endswith('usdt'):
            return text.upper()
        
        # Find in text
        for key, value in self.symbol_map.items():
            if key in text_lower:
                return value
        
        return None

    def extract_timeframe(self, text: str) -> str:
        text_lower = text.lower().strip()
        mapping = {
            '1m': '1 minute', '5m': '5 minutes', '15m': '15 minutes',
            '1h': '1 hour', '4h': '4 hours',
            'minute': '1 minute', 'minutes': '5 minutes',
            'hour': '1 hour', 'hours': '4 hours',
        }
        for key, value in mapping.items():
            if key in text_lower or value.replace(" ", "") in text_lower:
                return value
        return "1 hour"  # default
    
    def parse_prediction_query(self, query: str) -> Optional[Dict]:
        """Parse prediction-specific queries"""
        symbol = self.extract_symbol(query)
        timeframe = self.extract_timeframe(query)
        
        if not symbol:
            # Check conversation context
            if self.conversation_context['last_symbol']:
                symbol = self.conversation_context['last_symbol']
            else:
                return None
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'original_query': query
        }
    
    def make_prediction(self, symbol: str, timeframe: str) -> Dict:
        try:
            conn = psycopg2.connect(os.getenv("DATABASE_URL") + "?sslmode=require")
            db_tf = {
                "1 minute": "1minute", "5 minutes": "5minutes", "15 minutes": "15minutes",
                "1 hour": "1hour", "4 hours": "4hours", "1 day": "1day"
            }.get(timeframe, "1hour")

            # Determine timeframe from timestamp differences (smart auto-detect)
            query = """
                SELECT open, high, low, close, volume, timestamp
                FROM crypto_candles
                WHERE symbol = %s
                ORDER BY timestamp DESC
                LIMIT 300
            """
            df = pd.read_sql(query, conn, params=(symbol,))
            conn.close()

            if len(df) < 10:
                return {'success': False, 'error': 'Not enough data'}

            # Sort oldest first
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Auto-detect timeframe by looking at timestamp differences
            diffs = df['timestamp'].diff()[1:10]  # first 10 diffs
            minutes = diffs.dt.total_seconds().median() / 60

            if abs(minutes - 1) < 0.5:
                detected_tf = "1 minute"
            elif abs(minutes - 5) < 1:
                detected_tf = "5 minutes"
            elif abs(minutes - 15) < 2:
                detected_tf = "15 minutes"
            elif abs(minutes - 60) < 10:
                detected_tf = "1 hour"
            elif abs(minutes - 240) < 20:
                detected_tf = "4 hours"
            else:
                detected_tf = "1 hour"  # fallback

            # Use user-requested timeframe if available, otherwise use detected
            final_tf = timeframe if timeframe in ["1 minute", "5 minutes", "15 minutes", "1 hour", "4 hours"] else detected_tf

            # Resample to the requested timeframe (this is the key)
            df.set_index('timestamp', inplace=True)
            ohlc_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            df = df.resample({
                "1 minute": "1min",
                "5 minutes": "5min",
                "15 minutes": "15min",
                "1 hour": "1H",
                "4 hours": "4H",
                "1 day": "1D"
            }.get(final_tf, "1H")).apply(ohlc_dict).dropna()

            if len(df) < 100:
                return {'success': False, 'error': f'Not enough {final_tf} candles (need 100+)'}

            df = df.reset_index(drop=True)
            conn.close()

            if len(df) < 100:
                return {'success': False, 'error': 'Not enough data'}

            df = df.iloc[::-1].reset_index(drop=True)
            c = df['close'].astype(float)
            h = df['high'].astype(float)
            l = df['low'].astype(float)
            v = df['volume'].astype(float)

            # === PURE PANDAS INDICATORS (identical to TA-Lib) ===
            df['sma_20']      = c.rolling(20).mean()
            df['sma_50']      = c.rolling(50).mean()
            df['ema_12']      = c.ewm(span=12, adjust=False).mean()
            df['ema_26']      = c.ewm(span=26, adjust=False).mean()
            
            delta = c.diff()
            up = delta.clip(lower=0)
            down = (-delta).clip(lower=0)
            roll_up = up.ewm(com=13, adjust=False).mean()
            roll_down = down.ewm(com=13, adjust=False).mean()
            rs = roll_up / roll_down
            df['rsi'] = 100 - (100 / (1 + rs))
            
            df['macd']        = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist']   = df['macd'] - df['macd_signal']
            
            std20 = c.rolling(20).std()
            df['bb_middle']   = df['sma_20']
            df['bb_upper']    = df['sma_20'] + 2 * std20
            df['bb_lower']    = df['sma_20'] - 2 * std20
            
            tr = pd.DataFrame(index=df.index)
            tr['h_l']   = h - l
            tr['h_pc']  = abs(h - c.shift())
            tr['l_pc']  = abs(l - c.shift())
            tr['tr']    = tr.max(axis=1)
            df['atr']   = tr['tr'].rolling(14).mean()
            
            df['stoch_k'] = 100 * (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min() + 1e-8)
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
            df['willr']   = -100 * (h.rolling(14).max() - c) / (h.rolling(14).max() - l.rolling(14).min() + 1e-8)
            df['volume_sma'] = v.rolling(20).mean()
            df['roc']     = c.pct_change(10) * 100

            df = df.fillna(method='bfill').fillna(0)
            data = df[FEATURE_COLUMNS].tail(60).values.astype(float)

            result = predict_next_candle(symbol, timeframe, data)
            if "error" in result.lower() or "not trained" in result.lower():
                return {'success': False, 'error': result}

            pred_price = float(result.split("$")[1].replace(",", "").split(" ")[0])
            current = c.iloc[-1]

            return {
                'success': True,
                'symbol': symbol,
                'timeframe': timeframe,
                'prediction': {
                    'open':  round(current, 6),
                    'high':  round(pred_price * 1.002, 6),
                    'low':   round(pred_price * 0.998, 6),
                    'close': round(pred_price, 6),
                    'volume': int(v.mean() * 1.1)
                },
                'source': 'LSTM + Pure Pandas Indicators (TA-Lib free)'
            }

        except Exception as e:
            return {'success': False, 'error': f"Predict error: {str(e)}"}
         
    def process_query(self, query: str) -> Dict:
        """Main processing pipeline with conversational AI"""

        # === TOXICITY CHECK FIRST (highest priority) ===
        toxicity_response = self.handle_toxicity(query)
        if toxicity_response:
            return toxicity_response
        
        # Store query in context
        self.conversation_context['last_query'] = query
        
        # Detect intent
        intent = self.detect_intent(query)
        print(f"Detected intent: {intent}")
        
        # Handle different intents
        if intent == 'greeting':
            return {
                'success': True,
                'intent': 'greeting',
                'message': random.choice(self.greetings['responses'])
            }
        
        elif intent == 'gratitude':
            return {
                'success': True,
                'intent': 'gratitude',
                'message': random.choice(self.gratitude['responses'])
            }
        
        elif intent == 'wellbeing':
            return {
                'success': True,
                'intent': 'wellbeing',
                'message': random.choice(self.wellbeing['responses'])
            }
        
        elif intent == 'capabilities':
            return {
                'success': True,
                'intent': 'capabilities',
                'message': self.capabilities['responses'][0]
            }
        
        elif intent == 'farewell':
            return {
                'success': True,
                'intent': 'farewell',
                'message': random.choice(self.farewell['responses'])
            }
        
        elif intent == 'joke':
            return {
                'success': True,
                'intent': 'joke',
                'message': random.choice(self.jokes['responses'])
            }
        
        elif intent == 'technical_indicators':
            return {
                'success': True,
                'intent': 'technical_indicators',
                'message': self.get_technical_indicators_response(),
                'indicators_count': 22
            }
        
        elif intent == 'model_info':
            return {
                'success': True,
                'intent': 'model_info',
                'message': self.get_model_info_response()
            }
        
        elif intent == 'indicator_explain':
            explanation = self.explain_indicator(query)
            if explanation:
                return {
                    'success': True,
                    'intent': 'indicator_explain',
                    'message': explanation
                }
            else:
                return {
                    'success': True,
                    'intent': 'indicator_explain',
                    'message': "I'd be happy to explain any indicator! Try asking about: RSI, MACD, SMA, EMA, Bollinger Bands, ATR, Stochastic, or Williams %R. 📊"
                }
        
        elif intent == 'prediction':
            # Parse prediction query
            parsed = self.parse_prediction_query(query)
            
            if not parsed:
                return {
                    'success': False,
                    'intent': 'prediction',
                    'message': "I'd love to help with a prediction! 🔮 Which cryptocurrency are you interested in? I can predict BTC, ETH, SOL, ADA, XRP, BNB, and many more!"
                }
            
            # Make prediction
            result = self.make_prediction(parsed['symbol'], parsed['timeframe'])
            
            if result['success']:
                pred = result['prediction']
                
                # Format friendly message
                messages = [
                    f"🔮 **Prediction for {result['symbol']}** (next {result['timeframe']}):",
                    f"📊 **Prediction for {result['symbol']}** (timeframe: {result['timeframe']}):",
                    f"🚀 Here's what I see for **{result['symbol']}** in the next {result['timeframe']}:",
                ]
                
                return {
                    'success': True,
                    'intent': 'prediction',
                    'message': random.choice(messages),
                    'query': parsed,
                    'prediction': {
                        'symbol': result['symbol'],
                        'timeframe': result['timeframe'],
                        'open': f"${pred['open']:,.2f}",
                        'high': f"${pred['high']:,.2f}",
                        'low': f"${pred['low']:,.2f}",
                        'close': f"${pred['close']:,.2f}",
                        'volume': f"{pred['volume']:,.0f}"
                    },
                    'raw_prediction': pred,
                    'source': result['source']
                }
            else:
                return {
                    'success': False,
                    'intent': 'prediction',
                    'message': f"Oops! I encountered an issue getting the prediction: {result.get('error', 'Unknown error')} 😅"
                }
        
        else:  # unknown intent
            return {
                'success': False,
                'intent': 'unknown',
                'message': random.choice(self.confusion_responses)
            }


# Alias for backward compatibility
class CryptoPredictionNLP(JoAIConversationNLP):
    """Backward compatible alias"""
    pass


# Test function
def test_conversational_nlp():
    """Test the conversational AI"""
    nlp = JoAIConversationNLP()
    
    test_queries = [
        "hi",
        "what can you do?",
        "what indicators do you use?",
        "show me technical indicators",
        "list all features used",
        "what is RSI?",
        "explain MACD",
        "how accurate is your model?",
        "how does the LSTM work?",
        "predict SOL for 5 minutes",
        "thanks!",
        "goodbye",
    ]
    
    print("=" * 70)
    print("JoAI Conversational NLP Test")
    print("=" * 70)
    
    for query in test_queries:
        print(f"\n👤 User: '{query}'")
        result = nlp.process_query(query)
        message = result.get('message', 'No message')
        # Truncate long messages for display
        if len(message) > 500:
            message = message[:500] + "... [truncated]"
        print(f"🤖 JoAI: {message}")
        if result.get('prediction'):
            pred = result['prediction']
            print(f"   📊 Close: {pred['close']} | High: {pred['high']} | Low: {pred['low']}")
        print("-" * 70)


if __name__ == "__main__":
    test_conversational_nlp()