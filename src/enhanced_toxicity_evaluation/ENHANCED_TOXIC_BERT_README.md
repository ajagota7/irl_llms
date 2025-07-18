# Enhanced Toxic-BERT Category Analysis

This document describes the enhanced functionality for analyzing all Toxic-BERT categories in the Enhanced Toxicity Evaluation Pipeline.

## Overview

The enhanced analysis now supports comprehensive visualization and analysis of all Toxic-BERT categories:
- **toxic**: General toxicity
- **severe_toxic**: Severe toxicity
- **obscene**: Obscene language
- **threat**: Threatening language
- **insult**: Insulting language
- **identity_hate**: Identity-based hate speech

This provides much more granular insights into how different types of toxicity change during the detoxification process.

## New Features

### 1. Enhanced Visualization Manager

The `VisualizationManager` class now includes:

- **Automatic Detection**: Automatically detects Toxic-BERT categories from your data
- **Category-Specific Plots**: Individual scatter plots for each category
- **Category Heatmaps**: Visual comparison of model performance across all categories
- **Category Progression**: Training progression plots showing how each category improves over epochs
- **Category Delta Plots**: Distribution plots showing improvement patterns for each category

### 2. Standalone Enhanced Analysis Script

The `enhanced_wandb_analysis.py` script provides:

- **Complete Category Analysis**: Analyzes all Toxic-BERT categories automatically
- **Interactive Visualizations**: Creates comprehensive interactive plots
- **WandB Integration**: Logs all visualizations and metrics to WandB
- **Statistical Analysis**: Provides detailed statistics for each category

## Usage

### Option 1: Integrated with Main Pipeline

The enhanced functionality is automatically integrated into the main evaluation pipeline. When you run:

```bash
python run_evaluation.py
```

The visualization manager will automatically:
1. Detect Toxic-BERT categories in your data
2. Create category-specific visualizations
3. Log comprehensive metrics to WandB
4. Generate all plots and analysis

### Option 2: Standalone Analysis

For analyzing existing CSV files:

```bash
python enhanced_wandb_analysis.py
```

Or use the analysis script:

```bash
python analyze_results.py --csv-path your_results.csv
```

### Option 3: Test the Enhanced Functionality

Run the test script to see the enhanced analysis in action:

```bash
python test_enhanced_analysis.py
```

## Data Format

The enhanced analysis expects your CSV data to have columns in this format:

```
base_toxic_bert_toxic_score
base_toxic_bert_severe_toxic_score
base_toxic_bert_obscene_score
base_toxic_bert_threat_score
base_toxic_bert_insult_score
base_toxic_bert_identity_hate_score

detox_epoch_20_toxic_bert_toxic_score
detox_epoch_20_toxic_bert_severe_toxic_score
...
detox_epoch_100_toxic_bert_identity_hate_score
```

## Visualizations Created

### 1. Category-Specific Scatter Plots

For each Toxic-BERT category, creates scatter plots showing:
- Base model scores vs fine-tuned model scores
- Improvement metrics in hover tooltips
- Diagonal line showing "no change" baseline

### 2. Category Heatmap

Shows model performance across all categories:
- Rows: Models (epoch_20, epoch_40, etc.)
- Columns: Categories (toxic, severe_toxic, etc.)
- Colors: Improvement levels (green = better, red = worse)

### 3. Category Progression Plots

2x3 subplot showing training progression for all categories:
- X-axis: Training epochs
- Y-axis: Average improvement
- Each subplot: One category
- Zero line: No improvement baseline

### 4. Category Delta Distribution Plots

Histogram plots showing improvement distribution for each category:
- X-axis: Improvement amount
- Y-axis: Frequency
- Multiple models overlaid
- Zero line: No improvement baseline

## Metrics Logged

### Category-Specific Metrics

For each Toxic-BERT category, the system logs:
- `toxic_bert_{category}_overall_mean`: Average improvement across all models
- `toxic_bert_{category}_overall_std`: Standard deviation of improvements
- `toxic_bert_{category}_positive_rate`: Percentage of samples showing improvement
- `toxic_bert_{category}_best_model`: Best performing model for this category

### Summary Metrics

- `best_toxic_bert_category`: Category showing the most improvement
- `best_overall_model`: Best model across all categories
- `most_consistent_model`: Model with most consistent improvements

## Configuration

### Visualization Config

```yaml
visualization:
  create_plots: true
  save_plots: true
  plot_format: "html"
  include_toxic_bert_categories: true  # New option
```

### WandB Config

```yaml
logging:
  use_wandb: true
  wandb_project: "toxicity-evaluation"
  wandb_entity: "your_entity"
  wandb_tags: ["enhanced", "toxic_bert", "categories"]
```

## Example Output

When you run the enhanced analysis, you'll see:

```
🚀 STARTING ENHANCED WANDB ANALYSIS WITH TOXIC-BERT CATEGORIES
======================================================================
📊 Loading data...
✅ Loaded 1000 rows with 67 columns
📋 Detected models: ['base', 'detox_epoch_20', 'detox_epoch_40', 'detox_epoch_60', 'detox_epoch_80', 'detox_epoch_100']
🔍 Detected classifiers: ['roberta_toxicity', 'dynabench_hate', 'toxic_bert']
🎯 Detected Toxic-BERT categories: ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

📊 Calculating improvement deltas...
✅ Created delta_detox_epoch_20_vs_base_toxic_bert_toxic
✅ Created delta_detox_epoch_20_vs_base_toxic_bert_severe_toxic
✅ Created delta_detox_epoch_20_vs_base_toxic_bert_obscene
...

🔍 Creating Toxic-BERT category analysis...
📈 Creating Toxic-BERT category scatter plots...
🏆 Creating Toxic-BERT category heatmap...
📈 Creating Toxic-BERT category progression plots...
📊 Creating Toxic-BERT category delta plots...

✅ All enhanced visualizations created successfully!
🔗 View enhanced dashboard: https://wandb.ai/your_project/your_run
```

## Benefits

### 1. Granular Insights

Instead of just seeing overall toxicity reduction, you can now see:
- Which specific types of toxicity are most affected by detoxification
- Which categories are hardest to improve
- Whether certain categories improve faster than others

### 2. Better Model Understanding

The category analysis helps you understand:
- Model strengths and weaknesses across different toxicity types
- Whether detoxification is balanced across all categories
- Potential biases in the detoxification process

### 3. Improved Evaluation

With category-specific metrics, you can:
- Make more informed decisions about model selection
- Identify areas for improvement in your detoxification approach
- Better communicate results to stakeholders

## Troubleshooting

### No Toxic-BERT Categories Detected

If the system doesn't detect Toxic-BERT categories:

1. **Check Column Names**: Ensure your columns follow the pattern `{model}_toxic_bert_{category}_score`
2. **Verify Categories**: Make sure you have columns for at least some of: toxic, severe_toxic, obscene, threat, insult, identity_hate
3. **Check Data**: Ensure the columns contain valid numerical data

### Missing Visualizations

If some visualizations aren't created:

1. **Check Data Availability**: Ensure you have both base and fine-tuned model scores
2. **Verify Delta Columns**: Check that delta columns are being created correctly
3. **Review Logs**: Look for warning messages about missing columns

### WandB Issues

If WandB logging fails:

1. **Check Authentication**: Ensure your WandB API key is set correctly
2. **Verify Project**: Make sure the project exists and you have access
3. **Check Network**: Ensure you have internet connectivity

## Advanced Usage

### Custom Category Analysis

You can extend the analysis to include custom categories by modifying the `TOXIC_BERT_CATEGORIES` list in the code.

### Custom Visualizations

The modular design allows you to easily add new visualization types by extending the `VisualizationManager` class.

### Batch Analysis

For analyzing multiple datasets:

```python
from enhanced_wandb_analysis import run_complete_enhanced_analysis

csv_files = ["results1.csv", "results2.csv", "results3.csv"]
for csv_file in csv_files:
    df, run = run_complete_enhanced_analysis(csv_file)
    print(f"Completed analysis for {csv_file}")
```

## Conclusion

The enhanced Toxic-BERT category analysis provides unprecedented granularity in understanding how detoxification affects different types of toxicity. This enables more informed model development and evaluation, leading to better detoxification strategies and more effective AI safety measures. 