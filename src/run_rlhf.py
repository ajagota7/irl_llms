#!/usr/bin/env python3
"""
Simple script to run RLHF training with command-line arguments.
"""

import sys
import os

# Add the current directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rlhf_train import train_rlhf

if __name__ == "__main__":
    # Hydra will automatically handle command-line arguments
    train_rlhf() 