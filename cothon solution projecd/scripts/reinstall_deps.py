"""
Script to reinstall dependencies properly.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Reinstall dependencies properly."""
    print("Reinstalling dependencies...")
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Remove existing virtual environment if it exists
    venv_dir = project_root / "venv"
    if venv_dir.exists():
        print("Removing existing virtual environment...")
        if sys.platform == "win32":
            subprocess.run(["rmdir", "/s", "/q", str(venv_dir)], shell=True)
        else:
            subprocess.run(["rm", "-rf", str(venv_dir)])
    
    # Create new virtual environment
    print("Creating new virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", "venv"])
    
    # Activate virtual environment and install dependencies
    if sys.platform == "win32":
        python_path = venv_dir / "Scripts" / "python.exe"
        pip_path = venv_dir / "Scripts" / "pip.exe"
    else:
        python_path = venv_dir / "bin" / "python"
        pip_path = venv_dir / "bin" / "pip"
    
    # Upgrade pip
    print("Upgrading pip...")
    subprocess.run([str(pip_path), "install", "--upgrade", "pip"])
    
    # Install dependencies
    print("Installing dependencies...")
    subprocess.run([
        str(pip_path), "install", "-r", 
        str(project_root / "requirements.txt")
    ])
    
    # Download NLTK data
    print("Downloading NLTK data...")
    subprocess.run([
        str(python_path), "-c",
        "import nltk; nltk.download('punkt')"
    ])
    
    # Download spaCy model
    print("Downloading spaCy model...")
    subprocess.run([
        str(python_path), "-m", "spacy", "download", "en_core_web_sm"
    ])
    
    print("\nDependencies reinstalled successfully!")
    print("\nTo activate the virtual environment:")
    if sys.platform == "win32":
        print("    .\\venv\\Scripts\\activate")
    else:
        print("    source venv/bin/activate")

if __name__ == "__main__":
    main() 