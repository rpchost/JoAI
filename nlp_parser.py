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
            'UNI': ['uni', 'uniswap', 'uni/usdt', 'uniusdt']
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