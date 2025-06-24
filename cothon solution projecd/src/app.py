"""
Script to run the Legal Contract Analyzer web application.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from src.main import main

if __name__ == "__main__":
    main()