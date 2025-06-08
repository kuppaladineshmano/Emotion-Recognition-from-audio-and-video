# This is the entry point for Hugging Face Spaces
# It simply imports and runs the main application

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import and run the main application
from src.main import *