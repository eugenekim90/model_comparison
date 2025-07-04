#!/usr/bin/env python3
"""
Simple script to run the forecasting dashboard
"""

import subprocess
import sys
import os

def main():
    # Change to the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run the dashboard
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/dashboard.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 