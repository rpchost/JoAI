# import re
# from typing import Dict, Optional
# import requests

# class CryptoPredictionNLP:
#     """Enhanced NLP parser for crypto prediction queries"""
    
#     def __init__(self):
#         # Symbol mappings (handles variations)
#         self.symbol_map = {
#             # Bitcoin variations
#             'btc': 'BTCUSDT', 'bitcoin': 'BTCUSDT', 'btcusdt': 'BTCUSDT',
#             # Ethereum variations
#             'eth': 'ETHUSDT', 'ethereum': 'ETHUSDT', 'ethusdt': 'ETHUSDT',
#             # Binance Coin variations
#             'bnb': 'BNBUSDT', 'binance': 'BNBUSDT', 'bnbusdt': 'BNBUSDT',
#             # Cardano variations
#             'ada': 'ADAUSDT', 'cardano': 'ADAUSDT', 'adausdt': 'ADAUSDT',
#             # Solana variations
#             'sol': 'SOLUSDT', 'solana': 'SOLUSDT', 'solusdt': 'SOLUSDT',
#         }
        
#         # Timeframe mappings
#         self.timeframe_map = {
#             '1m': '1 minute', '5m': '5 minutes', '15m': '15 minutes', '30m': '30 minutes',
#             '1h': '1 hour', '4h': '4 hours', '1d': '1 day',
#             'minute': '1 minute', 'minutes': '5 minutes',
#             'hour': '1 hour', 'hours': '4 hours',
#             'day': '1 day', 'daily': '1 day',
#             'next hour': '1 hour', 'next day': '1 day',
#         }
        
#         # Query patterns (in order of specificity)
#         self.patterns = [
#             # Pattern 1: "predict SOLUSDT for 5 minutes"
#             r'predict\s+(\w+)\s+for\s+(\d+\s*\w+)',
            
#             # Pattern 2: "what is the prediction for BTC next hour"
#             r'(?:what\s+is\s+)?(?:the\s+)?prediction\s+for\s+(\w+)\s+(?:next\s+)?(\w+)',
            
#             # Pattern 3: "predict ETH 1h"
#             r'predict\s+(\w+)\s+(\d+[mhd])',
            
#             # Pattern 4: "BTC prediction for next hour"
#             r'(\w+)\s+prediction\s+(?:for\s+)?(?:next\s+)?(\w+)',
            
#             # Pattern 5: "what will BTC be in 1 hour"
#             r'what\s+will\s+(\w+)\s+be\s+in\s+(\d+\s*\w+)',
            
#             # Pattern 6: "SOLUSDT next 5 minutes"
#             r'(\w+)\s+(?:next\s+)?(\d+\s*\w+)',
            
#             # Pattern 7: "predict BTC"
#             r'predict\s+(\w+)',
            
#             # Pattern 8: "BTC prediction"
#             r'(\w+)\s+prediction',
            
#             # Pattern 9: Just symbol "SOLUSDT"
#             r'^(\w+)$',
#         ]

#     def extract_symbol(self, text: str) -> Optional[str]:
#         """Extract and normalize crypto symbol from text"""
#         text_lower = text.lower().strip()
        
#         # Direct match in symbol map
#         if text_lower in self.symbol_map:
#             return self.symbol_map[text_lower]
        
#         # Check if it already ends with USDT
#         if text_lower.endswith('usdt'):
#             return text.upper()
        
#         # Try to find any known symbol in the text
#         for key, value in self.symbol_map.items():
#             if key in text_lower:
#                 return value
        
#         return None

#     def extract_timeframe(self, text: str) -> str:
#         """Extract timeframe from text"""
#         text_lower = text.lower().strip()
        
#         # Direct match
#         if text_lower in self.timeframe_map:
#             return self.timeframe_map[text_lower]
        
#         # Check for patterns like "5 minutes", "1 hour"
#         time_match = re.search(r'(\d+)\s*(minute|minutes|hour|hours|day|days|m|h|d)', text_lower)
#         if time_match:
#             number = time_match.group(1)
#             unit = time_match.group(2)
            
#             # Normalize unit
#             if unit in ['m', 'minute']:
#                 return f"{number} minute" if number == '1' else f"{number} minutes"
#             elif unit in ['minutes']:
#                 return f"{number} minutes"
#             elif unit in ['h', 'hour']:
#                 return f"{number} hour" if number == '1' else f"{number} hours"
#             elif unit in ['hours']:
#                 return f"{number} hours"
#             elif unit in ['d', 'day', 'days']:
#                 return "1 day"
        
#         # Default
#         return "1 hour"

#     def parse_query(self, query: str) -> Dict:
#         """Parse natural language query into structured format"""
#         query_clean = query.strip().lower()
        
#         symbol = None
#         timeframe = None
        
#         # Try each pattern
#         for pattern in self.patterns:
#             match = re.search(pattern, query_clean, re.IGNORECASE)
#             if match:
#                 groups = match.groups()
                
#                 if len(groups) >= 1:
#                     symbol = self.extract_symbol(groups[0])
                
#                 if len(groups) >= 2 and groups[1]:
#                     timeframe = self.extract_timeframe(groups[1])
                
#                 if symbol:  # If we found a symbol, break
#                     break
        
#         # Default timeframe if not found
#         if not timeframe:
#             timeframe = "1 hour"
        
#         return {
#             'symbol': symbol,
#             'timeframe': timeframe,
#             'original_query': query
#         }

#     def process_query(self, query: str) -> Dict:
#         """Process query and return prediction"""
#         try:
#             # Parse the query
#             parsed = self.parse_query(query)
            
#             if not parsed['symbol']:
#                 return {
#                     'success': False,
#                     'message': "I couldn't understand your prediction request. Please try phrases like:\n"
#                                "- 'predict SOLUSDT for 5 minutes'\n"
#                                "- 'what is the prediction for BTC next hour'\n"
#                                "- 'predict ETH 1h'\n"
#                                "- 'SOL prediction'\n"
#                                "- 'BTCUSDT'",
#                     'query': parsed
#                 }
            
#             # Make prediction request (adjust URL to your actual endpoint)
#             try:
#                 import os
#                 base_url = os.getenv('BASE_URL', 'http://localhost:8081')
                
#                 response = requests.post(
#                     f"{base_url}/predict",
#                     json={
#                         'symbol': parsed['symbol'],
#                         'timeframe': '1h'  # You can map this from parsed timeframe
#                     },
#                     timeout=30
#                 )
                
#                 if response.status_code == 200:
#                     prediction_data = response.json()
                    
#                     return {
#                         'success': True,
#                         'message': f"Prediction for {parsed['symbol']} ({parsed['timeframe']}):",
#                         'query': parsed,
#                         'prediction': prediction_data.get('prediction', {}),
#                         'source': prediction_data.get('source', 'Unknown')
#                     }
#                 else:
#                     return {
#                         'success': False,
#                         'message': f"Failed to get prediction: {response.status_code}",
#                         'query': parsed
#                     }
                    
#             except requests.RequestException as e:
#                 return {
#                     'success': False,
#                     'message': f"Error fetching prediction: {str(e)}",
#                     'query': parsed
#                 }
                
#         except Exception as e:
#             return {
#                 'success': False,
#                 'message': f"Error processing query: {str(e)}",
#                 'error': str(e)
#             }


# # Convenience function
# def parse_crypto_query(query: str) -> Dict:
#     """Quick function to parse a crypto prediction query"""
#     parser = CryptoPredictionNLP()
#     return parser.process_query(query)


# # Test function
# def test_nlp_parser():
#     """Test various query formats"""
#     parser = CryptoPredictionNLP()
    
#     test_queries = [
#         "predict SOLUSDT for 5 minutes",
#         "what is the prediction for BTC next hour",
#         "predict ETH for 1 hour",
#         "SOL prediction",
#         "BTCUSDT",
#         "what will Solana be in 5 minutes",
#         "BTC next hour",
#         "predict cardano",
#         "ethereum 1h",
#     ]
    
#     print("=" * 60)
#     print("NLP Parser Test Results")
#     print("=" * 60)
    
#     for query in test_queries:
#         result = parser.parse_query(query)
#         print(f"\nQuery: '{query}'")
#         print(f"Symbol: {result['symbol']}")
#         print(f"Timeframe: {result['timeframe']}")
#         print("-" * 60)


# if __name__ == "__main__":
#     test_nlp_parser()

import re
import json
from typing import Dict, Optional, Tuple
from models.lstm_model import predict_next_candle

class CryptoPredictionNLP:
    """
    Natural Language Processing for crypto prediction queries
    Translates user phrases into API calls to the LSTM prediction model
    """

    def __init__(self, api_base_url: str = "http://localhost:8081"):
        self.api_base_url = api_base_url.rstrip('/')
        print(f"NLP initialized with API base URL: {self.api_base_url}")

        # Supported symbols and their patterns
        self.symbol_patterns = {
            'BTC': ['btc', 'bitcoin', 'btc/usdt', 'btcusdt'],
            'ETH': ['eth', 'ethereum', 'ether', 'eth/usdt', 'ethusdt'],
            'ADA': ['ada', 'cardano', 'ada/usdt', 'adausdt'],
            'DOT': ['dot', 'polkadot', 'dot/usdt', 'dotusdt'],
            'LINK': ['link', 'chainlink', 'link/usdt', 'linkusdt'],
            'UNI': ['uni', 'uniswap', 'uni/usdt', 'uniusdt'],
            'ADA': ['ADAUSDT', 'cardano', 'ADA/USDT', 'adausdt','ADA','ada'],
            'SOL': ['SOLUSDT', 'solana', 'solusdt','SOL/USDT','SOL','sol']
        }

        # Timeframe patterns
        self.timeframe_patterns = {
            '1m': ['1 minute', '1 min', '1m', 'one minute', 'next minute'],
            '5m': ['5 minutes', '5 mins', '5m', 'five minutes'],
            '15m': ['15 minutes', '15 mins', '15m', 'fifteen minutes', 'quarter hour'],
            '30m': ['30 minutes', '30 mins', '30m', 'thirty minutes', 'half hour'],
            '1h': ['1 hour', '1 hr', '1h', 'one hour', 'next hour', 'hourly'],
            '4h': ['4 hours', '4 hrs', '4h', 'four hours'],
            '1d': ['1 day', '1d', 'daily', 'one day', 'day'],
            '1w': ['1 week', '1w', 'weekly', 'one week', 'week']
        }

        # Keywords for prediction requests
        self.prediction_keywords = [
            'predict', 'prediction', 'forecast', 'what', 'tell me',
            'give me', 'show me', 'next', 'future', 'will be',
            'expect', 'anticipate', 'project'
        ]

    def parse_query(self, query: str) -> Optional[Dict]:
        """
        Parse natural language query and extract symbol/timeframe
        Returns dict with 'symbol' and 'timeframe' or None if not understood
        """
        query = query.lower().strip()

        # Check if it's a prediction request
        if not any(keyword in query for keyword in self.prediction_keywords):
            return None

        # Extract symbol
        symbol = self._extract_symbol(query)
        if not symbol:
            return None

        # Extract timeframe
        timeframe = self._extract_timeframe(query)
        if not timeframe:
            timeframe = '1h'  # Default to 1 hour

        return {
            'symbol': symbol,
            'timeframe': timeframe
        }

    def _extract_symbol(self, query: str) -> Optional[str]:
        """Extract cryptocurrency symbol from query"""
        for base_symbol, patterns in self.symbol_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    return f"{base_symbol}USDT"  # Default to USDT pair

        # Try to match any symbol-like pattern (e.g., BTC, ETH)
        symbol_match = re.search(r'\b([A-Z]{3,4})(?:/USDT|USDT)?\b', query.upper())
        if symbol_match:
            symbol = symbol_match.group(1)
            if symbol in self.symbol_patterns:
                return f"{symbol}USDT"

        return None

    def _extract_timeframe(self, query: str) -> Optional[str]:
        """Extract timeframe from query"""
        for tf, patterns in self.timeframe_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    return tf
        return None

    def make_prediction_request(self, parsed_query: Dict) -> Dict:
        """
        Call the real prediction API
        Returns the prediction response
        """
        print(f"NLP calling real prediction API for: {parsed_query}")

        try:
            symbol = parsed_query['symbol']
            timeframe = parsed_query['timeframe']

            # Call the real prediction function
            prediction = predict_next_candle(symbol)

            # Format response to match API structure
            result = {
                "symbol": symbol,
                "timeframe": timeframe,
                "prediction": {
                    "open": round(prediction["open"], 2),
                    "high": round(prediction["high"], 2),
                    "low": round(prediction["low"], 2),
                    "close": round(prediction["close"], 2),
                    "volume": round(prediction["volume"], 2)
                },
                "source": "MySQL + LSTM Model + Technical Indicators"
            }

            print(f"NLP real prediction success: {result}")
            return result

        except Exception as e:
            print(f"NLP prediction error: {e}")
            raise Exception(f"Prediction error: {str(e)}")

    def process_query(self, query: str) -> Dict:
        """
        Complete pipeline: parse query -> make API call -> return formatted response
        """
        print(f"NLP processing query: '{query}'")

        # Parse the natural language query
        parsed = self.parse_query(query)
        print(f"NLP parsed query: {parsed}")

        if not parsed:
            print("NLP: Could not parse query")
            return {
                'success': False,
                'message': "I couldn't understand your prediction request. Please try phrases like 'what is the prediction for BTC next hour' or 'predict ETH for 5 minutes'."
            }

        try:
            print("NLP: Making prediction request")
            # Make the prediction request
            prediction_result = self.make_prediction_request(parsed)
            print("NLP: Got prediction result")

            # Format the response for the user
            response = self._format_response(parsed, prediction_result)
            print("NLP: Formatted response")
            return response

        except Exception as e:
            print(f"NLP: Error in process_query: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'message': f"Sorry, I encountered an error getting the prediction: {str(e)}"
            }

    def _format_response(self, parsed_query: Dict, prediction_result: Dict) -> Dict:
        """Format the prediction result into a user-friendly response"""
        symbol = parsed_query['symbol']
        timeframe = parsed_query['timeframe']

        prediction = prediction_result['prediction']
        source = prediction_result['source']

        # Format timeframe for display
        timeframe_display = {
            '1m': '1 minute',
            '5m': '5 minutes',
            '15m': '15 minutes',
            '30m': '30 minutes',
            '1h': '1 hour',
            '4h': '4 hours',
            '1d': '1 day',
            '1w': '1 week'
        }.get(timeframe, timeframe)

        response = {
            'success': True,
            'query': parsed_query,
            'prediction': {
                'symbol': symbol,
                'timeframe': timeframe_display,
                'open': f"${prediction['open']:,.2f}",
                'high': f"${prediction['high']:,.2f}",
                'low': f"${prediction['low']:,.2f}",
                'close': f"${prediction['close']:,.2f}",
                'volume': f"{prediction['volume']:,.0f}"
            },
            'source': source,
            'message': f"Here's my prediction for {symbol} over the next {timeframe_display}:"
        }

        return response

# Example usage
if __name__ == "__main__":
    nlp = CryptoPredictionNLP()

    # Test queries
    test_queries = [
        "what is the prediction for the next hour for BTCUSDT",
        "predict ETH for 5 minutes",
        "what will bitcoin be in 1 hour",
        "give me the forecast for ADA next day",
        "show me DOT prediction for 4 hours"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        result = nlp.process_query(query)
        if result['success']:
            pred = result['prediction']
            print(f"✅ {result['message']}")
            print(f"   Symbol: {pred['symbol']} | Timeframe: {pred['timeframe']}")
            print(f"   Predicted Close: {pred['close']}")
        else:
            print(f"❌ {result['message']}")