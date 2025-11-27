# update_data_local.py — Local Windows Script for Data Updates
"""
Run this script on your local machine via Task Scheduler.
It connects to your Render PostgreSQL database and updates crypto data.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

def main():
    print("=" * 70)
    print(f"JoAI Data Update - Started at {datetime.now()}")
    print("=" * 70)
    
    # Verify DATABASE_URL exists
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not found in .env file!")
        print(f"Looking for .env at: {env_path}")
        return 1
    
    print(f"✓ Database URL loaded (first 30 chars): {db_url[:30]}...")
    
    try:
        # Import and run data fetch
        print("\nFetching data from Binance...")
        from fetch_data import populate_multiple_symbols
        
        populate_multiple_symbols()
        
        print("\n" + "=" * 70)
        print(f"✓ SUCCESS - Data update completed at {datetime.now()}")
        print("=" * 70)
        return 0
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"✗ ERROR: {str(e)}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    
    # Keep window open for 5 seconds to see output
    import time
    time.sleep(5)
    
    sys.exit(exit_code)