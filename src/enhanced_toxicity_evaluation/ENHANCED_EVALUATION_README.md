# Enhanced Toxicity Evaluation Pipeline

## 🚀 Overview

This enhanced version of the toxicity evaluation pipeline provides comprehensive analysis capabilities with:

- **More Prompts**: Increased from 1000 to 5000 prompts for better statistical power
- **Scatter Plots**: Comprehensive visualization suite including scatter plots, distributions, and comparisons
- **Better Dataset Saving**: Multiple format exports (CSV, Parquet, JSON) with comprehensive metadata
- **HTML Reports**: Interactive HTML reports with all visualizations
- **Enhanced Metrics**: Statistical significance testing and effect size calculations

## 📊 Key Improvements

### 1. Increased Dataset Size
- **Before**: 1000 prompts with toxicity threshold > 0.4
- **After**: 5000 prompts with toxicity threshold > 0.3
- **Benefit**: Better statistical power and more diverse prompt distribution

### 2. Comprehensive Visualizations
- **Scatter Plots**: Model comparison plots with regression lines and improvement metrics
- **Distribution Plots**: Histograms and KDE plots for toxicity score distributions
- **Correlation Matrix**: Heatmap showing relationships between different classifiers
- **Progression Plots**: Epoch-by-epoch analysis for training progression
- **Comparison Plots**: Improvement/regression analysis with statistical significance

### 3. Enhanced Data Export
- **Multiple Formats**: CSV, Parquet, JSON exports
- **Comprehensive Metadata**: Experiment configuration, prompt analysis, summary statistics
- **Structured Organization**: Clear directory structure with README files
- **Reproducibility**: Complete experiment snapshots for reproducibility

## 🛠️ Usage

### Basic Enhanced Evaluation

```bash
# Run the enhanced evaluation with all improvements
python run_enhanced_evaluation.py \
  models=pythia_70m_detox_all_epochs \
  generation.max_new_tokens=20 \
  generation.temperature=0.3 \
  generation.do_sample=false \
  experiment.name="pythia_70m_detox_progression_enhanced"
```

### Colab Command

```python
!python run_enhanced_evaluation.py \
  models=pythia_70m_detox_all_epochs \
  generation.max_new_tokens=20 \
  generation.temperature=0.3 \
  generation.do_sample=false \
  experiment.name="pythia_70m_detox_progression_enhanced"
```

### Custom Configuration

```bash
# Use more prompts with custom filtering
python run_enhanced_evaluation.py \
  dataset.dataset.max_prompts=10000 \
  dataset.filtering.toxicity_threshold=0.25 \
  dataset.filtering.strategy="range" \
  dataset.filtering.min_toxicity=0.2 \
  dataset.filtering.max_toxicity=0.8 \
  experiment.name="large_scale_evaluation"

# Disable certain plot types
python run_enhanced_evaluation.py \
  visualization.plot_types.progression_plots=false \
  visualization.plot_types.correlation_matrix=false \
  experiment.name="focused_evaluation"
```

## 📁 Output Structure

```
results/
└── pythia_70m_detox_progression_enhanced/
    ├── toxicity_evaluation_results.csv          # Main results
    ├── evaluation_summary.json                  # Summary metrics
    ├── raw_toxicity_results.json               # Raw classifier outputs
    ├── evaluation_report.html                  # Interactive HTML report
    │
    ├── plots/                                  # All visualizations
    │   ├── scatter_comparison_roberta_toxicity.png
    │   ├── scatter_comparison_toxic_bert.png
    │   ├── distribution_base.png
    │   ├── distribution_detox_epoch_20.png
    │   ├── correlation_matrix.png
    │   ├── progression_roberta_toxicity.png
    │   ├── improvement_roberta_toxicity.png
    │   └── improvement_summary.png
    │
    ├── dataset_exports/                        # Comprehensive data exports
    │   ├── toxicity_evaluation_results.csv
    │   ├── toxicity_evaluation_results.parquet
    │   ├── toxicity_evaluation_results.json
    │   ├── summary_statistics.json
    │   ├── model_comparisons.csv
    │   ├── model_comparisons.json
    │   ├── prompt_analysis.json
    │   ├── experiment_config.json
    │   └── README.md
    │
    ├── reports/                                # Additional reports
    └── logs/                                   # Execution logs
        └── evaluation.log
```

## 📈 Visualization Types

### 1. Scatter Plots
- **Purpose**: Compare models against baseline
- **Features**: 
  - Regression lines with R² values
  - Diagonal line (y=x) for reference
  - Improvement metrics in titles
  - Equal aspect ratio for fair comparison

### 2. Distribution Plots
- **Purpose**: Show toxicity score distributions
- **Features**:
  - Histograms with KDE overlays
  - Mean and median lines
  - Model-wise and classifier-wise grouping
  - Statistical annotations

### 3. Correlation Matrix
- **Purpose**: Show relationships between classifiers
- **Features**:
  - Annotated correlation values
  - Color-coded heatmap
  - Upper triangle masking
  - Statistical significance indicators

### 4. Progression Plots
- **Purpose**: Show training progression across epochs
- **Features**:
  - Epoch-by-epoch histograms
  - Statistical summaries
  - Trend analysis
  - Improvement tracking

### 5. Comparison Plots
- **Purpose**: Analyze improvements vs regressions
- **Features**:
  - Zero-line reference
  - Improvement rate calculations
  - Statistical significance testing
  - Effect size measurements

## 📊 Enhanced Metrics

### Statistical Significance
- **Wilcoxon Signed-Rank Test**: For paired comparisons
- **Paired T-Test**: For normally distributed data
- **Mann-Whitney U Test**: For independent samples
- **Multiple Comparison Correction**: Bonferroni, Holm, FDR

### Effect Sizes
- **Cohen's d**: Standardized mean difference
- **Improvement Rate**: Percentage of improved samples
- **Mean Improvement**: Average toxicity reduction
- **Confidence Intervals**: 95% CI for estimates

### Distribution Metrics
- **Percentiles**: 25th, 50th, 75th, 95th
- **Skewness & Kurtosis**: Distribution shape
- **Threshold Rates**: High/medium/low toxicity rates
- **Outlier Detection**: Statistical outlier identification

## 🔧 Configuration Options

### Dataset Configuration
```yaml
dataset:
  dataset:
    max_prompts: 5000  # Number of prompts to use
    shuffle_seed: 42   # Random seed for reproducibility
  
  filtering:
    strategy: "toxicity_threshold"  # or "range", "top_k", "random"
    toxicity_threshold: 0.3         # Minimum toxicity for inclusion
    metrics: ["toxicity", "severe_toxicity"]
```

### Visualization Configuration
```yaml
visualization:
  enabled: true
  
  plot_types:
    scatter_plots: true
    distribution_plots: true
    comparison_plots: true
    correlation_matrix: true
    progression_plots: true
    summary_plots: true
  
  plot_settings:
    figure_size: [12, 8]
    dpi: 300
    format: "png"
```

### Output Configuration
```yaml
output:
  local:
    save_csv: true
    save_json: true
    save_plots: true
    save_reports: true
```

## 📈 Performance Considerations

### Memory Usage
- **Large Datasets**: 5000 prompts × 6 models × 2 classifiers ≈ 60K evaluations
- **Recommendation**: Use batch processing and cleanup
- **Monitoring**: Check memory usage during execution

### Computation Time
- **Model Loading**: ~30-60 seconds per model
- **Generation**: ~2-5 seconds per prompt (depending on max_new_tokens)
- **Classification**: ~1-2 seconds per batch
- **Visualization**: ~30-60 seconds total

### Optimization Tips
1. **Use Caching**: Enable dataset caching to avoid re-downloading
2. **Batch Processing**: Optimize batch sizes for your hardware
3. **Parallel Processing**: Use multiple workers for classification
4. **Memory Management**: Monitor and cleanup unused models

## 🔍 Analysis Examples

### Load Results
```python
import pandas as pd

# Load main results
df = pd.read_csv("results/pythia_70m_detox_progression_enhanced/toxicity_evaluation_results.csv")

# Load summary statistics
import json
with open("results/pythia_70m_detox_progression_enhanced/dataset_exports/summary_statistics.json") as f:
    stats = json.load(f)
```

### Analyze Improvements
```python
# Find best improvements
delta_cols = [col for col in df.columns if col.startswith('delta_')]
best_improvements = df.nlargest(10, 'delta_detox_epoch_100_vs_base_roberta_toxicity_score')

# Calculate overall improvement
improvement_summary = {}
for col in delta_cols:
    deltas = df[col].dropna()
    improvement_summary[col] = {
        'mean_improvement': deltas.mean(),
        'improved_rate': (deltas > 0).mean(),
        'std_improvement': deltas.std()
    }
```

### Create Custom Plots
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Custom scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(df['output_base_roberta_toxicity_score'], 
           df['output_detox_epoch_100_roberta_toxicity_score'], 
           alpha=0.6)
plt.plot([0, 1], [0, 1], 'r--', label='No Change')
plt.xlabel('Base Model Toxicity')
plt.ylabel('Detoxified Model Toxicity')
plt.title('Detoxification Effect')
plt.legend()
plt.show()
```

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Reduce batch size
   python run_enhanced_evaluation.py generation.batch_size=4
   
   # Use fewer prompts
   python run_enhanced_evaluation.py dataset.dataset.max_prompts=1000
   ```

2. **Slow Generation**
   ```bash
   # Reduce max tokens
   python run_enhanced_evaluation.py generation.max_new_tokens=10
   
   # Use deterministic generation
   python run_enhanced_evaluation.py generation.do_sample=false
   ```

3. **Plot Generation Errors**
   ```bash
   # Disable problematic plots
   python run_enhanced_evaluation.py \
     visualization.plot_types.correlation_matrix=false \
     visualization.plot_types.progression_plots=false
   ```

### Debug Mode
```bash
# Enable detailed logging
python run_enhanced_evaluation.py logging.log_level=DEBUG

# Run minimal evaluation
python run_enhanced_evaluation.py \
  dataset.dataset.max_prompts=10 \
  visualization.plot_types.scatter_plots=false
```

## 📚 Citation

If you use this enhanced evaluation pipeline, please cite:

```bibtex
@misc{enhanced_toxicity_evaluation_2024,
  title={Enhanced Toxicity Evaluation Pipeline for Language Models},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/enhanced-toxicity-evaluation}
}
```

## 🤝 Contributing

To contribute to the enhanced evaluation pipeline:

1. **Fork the repository**
2. **Create a feature branch**
3. **Add tests for new functionality**
4. **Update documentation**
5. **Submit a pull request**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 