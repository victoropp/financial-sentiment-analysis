"""
Financial Sentiment Analysis - Streamlit Entry Point
"""
import os
import sys
from pathlib import Path

# Get the directory containing this file (project root)
PROJECT_ROOT = Path(__file__).resolve().parent

# Change to project root directory
os.chdir(PROJECT_ROOT)

# Add src and deployment to path
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'deployment'))

# Run the main Streamlit app by importing it
exec(compile(open(PROJECT_ROOT / "deployment" / "app.py").read(), PROJECT_ROOT / "deployment" / "app.py", 'exec'))
