#!/usr/bin/env python3
"""Test script for resume functionality."""

import sys
from pathlib import Path
from main import run_training

def test_resume():
    """Test the resume functionality."""
    print("Testing resume functionality...")
    
    # First, run a short training to create checkpoints
    print("\n1. Running initial training for 2 epochs...")
    try:
        run_training(
            config_path="config_test_resume.yaml",
            resume=False
        )
        print("✓ Initial training completed")
    except Exception as e:
        print(f"✗ Initial training failed: {e}")
        return False
    
    # Then test resume
    print("\n2. Testing resume from checkpoint...")
    try:
        run_training(
            config_path="config_test_resume.yaml", 
            resume=True
        )
        print("✓ Resume training completed")
        return True
    except Exception as e:
        print(f"✗ Resume training failed: {e}")
        return False

if __name__ == "__main__":
    success = test_resume()
    sys.exit(0 if success else 1)