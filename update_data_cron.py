# update_data_cron.py — Render Cron Job Script
"""
This script fetches fresh crypto data and stores it in PostgreSQL.
Designed to run as a Render Cron Job every 30 minutes.
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print(f"=== JoAI Data Update Started at {datetime.utcnow()} UTC ===")
    
    try:
        # Import fetch function
        from fetch_data import populate_multiple_symbols
        
        # Fetch data for all coins
        print("Fetching data for all cryptocurrencies...")
        populate_multiple_symbols()
        
        print(f"✓ Data update completed successfully at {datetime.utcnow()} UTC")
        return 0
        
    except Exception as e:
        print(f"✗ Error during data update: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)