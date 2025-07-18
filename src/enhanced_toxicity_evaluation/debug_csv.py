#!/usr/bin/env python3
"""
Debug script to examine CSV file structure and column naming patterns.
"""

import pandas as pd
import sys
from pathlib import Path

def analyze_csv_structure(csv_path: str):
    """Analyze the structure of a CSV file."""
    print(f"📊 Analyzing CSV file: {csv_path}")
    print("=" * 60)
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    print(f"📋 Basic Info:")
    print(f"  - Rows: {len(df)}")
    print(f"  - Columns: {len(df.columns)}")
    print(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n📝 Column Analysis:")
    
    # Categorize columns
    output_cols = [col for col in df.columns if col.startswith('output_') and not col.endswith('_score')]
    score_cols = [col for col in df.columns if col.endswith('_score')]
    delta_cols = [col for col in df.columns if col.startswith('delta_')]
    prompt_cols = [col for col in df.columns if col.startswith('prompt_')]
    full_cols = [col for col in df.columns if col.startswith('full_')]
    other_cols = [col for col in df.columns if col not in output_cols + score_cols + delta_cols + prompt_cols + full_cols]
    
    print(f"  - Output columns: {len(output_cols)}")
    print(f"  - Score columns: {len(score_cols)}")
    print(f"  - Delta columns: {len(delta_cols)}")
    print(f"  - Prompt columns: {len(prompt_cols)}")
    print(f"  - Full text columns: {len(full_cols)}")
    print(f"  - Other columns: {len(other_cols)}")
    
    print(f"\n🔍 Output Columns (models):")
    for col in output_cols:
        model_name = col.replace('output_', '')
        print(f"  - {model_name}")
    
    print(f"\n📊 Score Columns:")
    for col in score_cols:
        print(f"  - {col}")
    
    print(f"\n📈 Delta Columns:")
    for col in delta_cols:
        print(f"  - {col}")
    
    print(f"\n📝 Prompt Columns:")
    for col in prompt_cols:
        print(f"  - {col}")
    
    print(f"\n📄 Full Text Columns:")
    for col in full_cols:
        print(f"  - {col}")
    
    print(f"\n🔧 Other Columns:")
    for col in other_cols:
        print(f"  - {col}")
    
    # Analyze model naming patterns
    print(f"\n🎯 Model Naming Pattern Analysis:")
    model_names = [col.replace('output_', '') for col in output_cols]
    
    # Check for epoch patterns
    epoch_models = []
    check_models = []
    step_models = []
    number_models = []
    
    import re
    for model_name in model_names:
        if 'epoch_' in model_name:
            epoch_models.append(model_name)
        elif 'check_' in model_name:
            check_models.append(model_name)
        elif 'step_' in model_name:
            step_models.append(model_name)
        elif re.search(r'\d+', model_name):
            number_models.append(model_name)
    
    if epoch_models:
        print(f"  - Epoch models: {epoch_models}")
    if check_models:
        print(f"  - Checkpoint models: {check_models}")
    if step_models:
        print(f"  - Step models: {step_models}")
    if number_models:
        print(f"  - Models with numbers: {number_models}")
    
    # Sample data
    print(f"\n📋 Sample Data (first 3 rows):")
    print(df.head(3).to_string())
    
    # Check for missing values
    print(f"\n❓ Missing Values Analysis:")
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    if len(missing_cols) > 0:
        print("  Columns with missing values:")
        for col, count in missing_cols.items():
            print(f"    - {col}: {count} missing values")
    else:
        print("  No missing values found!")
    
    return df

def main():
    if len(sys.argv) != 2:
        print("Usage: python debug_csv.py <path_to_csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    try:
        df = analyze_csv_structure(csv_path)
        print(f"\n✅ Analysis completed successfully!")
    except Exception as e:
        print(f"❌ Error analyzing CSV: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 