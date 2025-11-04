"""
WSGI configuration for PythonAnywhere deployment
"""
import os
import sys

# Add project directory to path
project_home = '/home/rpchost/JoAI'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Set up environment variables
os.environ.setdefault('DB_CONNECTION', 'mysql')
os.environ.setdefault('MYSQL_HOST', 'rpchost.mysql.pythonanywhere-services.com')
os.environ.setdefault('MYSQL_PORT', '3306')
os.environ.setdefault('MYSQL_DATABASE', 'rpchost$joai_db')
os.environ.setdefault('MYSQL_USER', 'rpchost')
os.environ.setdefault('MYSQL_PASSWORD', 'your_mysql_password_here')
os.environ.setdefault('MYSQL_CHARSET', 'utf8mb4')

# Import Flask app
from main import app

# For PythonAnywhere, we need to expose the app
application = app