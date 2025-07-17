#!/usr/bin/env python3
"""
Check and display results from the evaluation pipeline.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

def check_results():
    """Check for evaluation results and display them."""
    print("🔍 Checking for evaluation results...")
    
    # Look for result directories
    possible_dirs = [
        "test_results",
        "debug_results", 
        "results",
        "output"
    ]
    
    found_results = False
    
    for dir_name in possible_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Found results directory: {dir_name}")
            found_results = True
            
            # List files in the directory
            dir_path = Path(dir_name)
            files = list(dir_path.glob("*"))
            
            if files:
                print(f"📁 Files in {dir_name}:")
                for file in files:
                    print(f"  - {file.name}")
                    
                    # Try to read and display content
                    if file.suffix == '.csv':
                        try:
                            df = pd.read_csv(file)
                            print(f"    📊 CSV shape: {df.shape}")
                            print(f"    📋 Columns: {list(df.columns)}")
                            
                            # Show toxicity score columns
                            toxicity_cols = [col for col in df.columns if col.endswith('_score')]
                            if toxicity_cols:
                                print(f"    🎯 Toxicity columns: {toxicity_cols}")
                                for col in toxicity_cols:
                                    scores = df[col].dropna()
                                    if len(scores) > 0:
                                        print(f"      {col}: mean={scores.mean():.4f}, std={scores.std():.4f}")
                        except Exception as e:
                            print(f"    ❌ Error reading CSV: {e}")
                    
                    elif file.suffix == '.json':
                        try:
                            with open(file, 'r') as f:
                                data = json.load(f)
                            print(f"    📄 JSON keys: {list(data.keys())}")
                            
                            # Show comparison metrics if available
                            if 'comparison_metrics' in data:
                                comparison = data['comparison_metrics']
                                if comparison:
                                    print(f"    📊 Comparison results:")
                                    for model_name, classifier_results in comparison.items():
                                        print(f"      {model_name}:")
                                        for classifier_name, metrics in classifier_results.items():
                                            improvement = metrics.get('improvement', 0)
                                            improved_rate = metrics.get('improved_rate', 0)
                                            print(f"        {classifier_name}: improvement={improvement:.4f}, improved_rate={improved_rate:.2%}")
                        except Exception as e:
                            print(f"    ❌ Error reading JSON: {e}")
            else:
                print(f"    📭 Directory is empty")
    
    if not found_results:
        print("❌ No results directories found")
        print("💡 Try running the evaluation pipeline first")
    
    return found_results

if __name__ == "__main__":
    check_results() 