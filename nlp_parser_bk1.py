import re
import random
from typing import Dict, Optional, List
from datetime import datetime
from models.lstm_model import predict_next_candle

class JoAIConversationNLP:
    """
    Human-like conversational AI for crypto predictions
    Handles greetings, small talk, context, and predictions
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
        }
        
        # Timeframe patterns
        self.timeframe_map = {
            '1m': '1 minute', '5m': '5 minutes', '15m': '15 minutes', '30m': '30 minutes',
            '1h': '1 hour', '4h': '4 hours', '1d': '1 day',
            'minute': '1 minute', 'minutes': '5 minutes',
            'hour': '1 hour', 'hours': '4 hours',
            'day': '1 day', 'daily': '1 day',
        }
        
        # Greeting patterns
        self.greetings = {
            'patterns': [
                r'\b(hi|hello|hey|greetings|good\s+(morning|afternoon|evening|day)|howdy|yo|sup|what\'s up)\b',
            ],
            'responses': [
                "Hi there! 👋 I'm JoAI, your crypto prediction assistant. How can I help you today?",
                "Hello! 😊 I'm here to help you with crypto predictions. What would you like to know?",
                "Hey! Great to see you! I can predict crypto prices for BTC, ETH, SOL, ADA, and BNB. What interests you?",
                "Hi! 🚀 Ready to explore crypto predictions together? Just ask me about any cryptocurrency!",
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
            'patterns': [
                r'\b(what can you do|what do you do|help|capabilities|features|how does this work|how to use)\b',
            ],
            'responses': [
                """I'm JoAI, your crypto prediction AI! 🤖 Here's what I can do:

📊 **Crypto Predictions**: I use advanced LSTM neural networks to predict prices for:
   • Bitcoin (BTC)
   • Ethereum (ETH)
   • Solana (SOL)
   • Cardano (ADA)
   • Binance Coin (BNB)

🎯 **How to ask**:
   • "Predict BTC for next hour"
   • "What will SOL be in 5 minutes?"
   • "ETH prediction"
   • "Show me Bitcoin forecast"

💡 **Just ask naturally!** I understand different ways of asking. Try it out!""",
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
            "Hmm, I'm not quite sure what you mean. 🤔 Could you rephrase that? Or try asking about a crypto prediction!",
            "I didn't quite catch that. Try asking something like 'predict BTC for 1 hour' or 'what will ETH be?'",
            "I'm a bit confused! 😅 I'm best at crypto predictions. Try asking about Bitcoin, Ethereum, Solana, Cardano, or BNB!",
        ]

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
        
        # Check for prediction request
        prediction_keywords = ['predict', 'prediction', 'forecast', 'what will', 'price', 'value', 'worth', 'next']
        if any(keyword in query_lower for keyword in prediction_keywords):
            return 'prediction'
        
        # Check if just a symbol mentioned
        if self.extract_symbol(query):
            return 'prediction'
        
        return 'unknown'

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
        """Extract timeframe from text"""
        text_lower = text.lower().strip()
        
        # Direct match
        if text_lower in self.timeframe_map:
            return self.timeframe_map[text_lower]
        
        # Pattern matching
        time_match = re.search(r'(\d+)\s*(minute|minutes|hour|hours|day|days|m|h|d)', text_lower)
        if time_match:
            number = time_match.group(1)
            unit = time_match.group(2)
            
            if unit in ['m', 'minute']:
                return f"{number} minute" if number == '1' else f"{number} minutes"
            elif unit == 'minutes':
                return f"{number} minutes"
            elif unit in ['h', 'hour']:
                return f"{number} hour" if number == '1' else f"{number} hours"
            elif unit == 'hours':
                return f"{number} hours"
            elif unit in ['d', 'day', 'days']:
                return "1 day"
        
        return "1 hour"

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
        """Make prediction using LSTM model"""
        try:
            prediction = predict_next_candle(symbol)
            
            # Store in context
            self.conversation_context['last_symbol'] = symbol
            self.conversation_context['last_prediction'] = prediction
            
            return {
                'success': True,
                'symbol': symbol,
                'timeframe': timeframe,
                'prediction': {
                    'open': round(prediction['open'], 2),
                    'high': round(prediction['high'], 2),
                    'low': round(prediction['low'], 2),
                    'close': round(prediction['close'], 2),
                    'volume': round(prediction['volume'], 2)
                },
                'source': 'LSTM Neural Network + Technical Indicators'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def process_query(self, query: str) -> Dict:
        """Main processing pipeline with conversational AI"""
        
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
        
        elif intent == 'prediction':
            # Parse prediction query
            parsed = self.parse_prediction_query(query)
            
            if not parsed:
                return {
                    'success': False,
                    'intent': 'prediction',
                    'message': "I'd love to help with a prediction! 🔮 Which cryptocurrency are you interested in? I can predict BTC, ETH, SOL, ADA, or BNB!"
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
        "hello there!",
        "how are you?",
        "what can you do?",
        "predict SOL for 5 minutes",
        "what will BTC be next hour?",
        "ETH prediction",
        "thanks!",
        "thank you so much",
        "tell me a joke",
        "goodbye",
        "what is the meaning of life?",  # Unknown intent
    ]
    
    print("=" * 70)
    print("JoAI Conversational NLP Test")
    print("=" * 70)
    
    for query in test_queries:
        print(f"\n👤 User: '{query}'")
        result = nlp.process_query(query)
        print(f"🤖 JoAI: {result.get('message', 'No message')}")
        if result.get('prediction'):
            pred = result['prediction']
            print(f"   📊 Close: {pred['close']} | High: {pred['high']} | Low: {pred['low']}")
        print("-" * 70)


if __name__ == "__main__":
    test_conversational_nlp()