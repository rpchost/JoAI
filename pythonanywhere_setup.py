#!/usr/bin/env python3
"""
Setup script for PythonAnywhere deployment
Run this in PythonAnywhere console to set up the database
"""
import os
import sys

# Add project directory to path
project_home = '/home/rpchost/JoAI'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

def main():
    print("Setting up JoAI on PythonAnywhere...")

    # Set environment variables for PythonAnywhere
    os.environ['DB_CONNECTION'] = 'mysql'
    os.environ['MYSQL_HOST'] = 'rpchost.mysql.pythonanywhere-services.com'
    os.environ['MYSQL_PORT'] = '3306'
    os.environ['MYSQL_DATABASE'] = 'rpchost$joai_db'
    os.environ['MYSQL_USER'] = 'rpchost'
    os.environ['MYSQL_PASSWORD'] = input("Enter your MySQL password: ")
    os.environ['MYSQL_CHARSET'] = 'utf8mb4'

    try:
        # Run database initialization
        print("Initializing database...")
        exec(open('init_mysql.py').read())

        # Test database connection
        print("Testing database connection...")
        exec(open('test_db.py').read())

        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Upload your trained model files (models/saved_model.keras, models/saved_model_scaler.pkl, models/saved_model_target_scaler.pkl)")
        print("2. Update pythonanywhere_wsgi.py with your actual MySQL password")
        print("3. Reload your web app in PythonAnywhere dashboard")
        print("4. Test the endpoints: https://rpchost.pythonanywhere.com/")

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()