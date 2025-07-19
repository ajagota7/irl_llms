# Enhanced Toxicity Evaluation - Visualization Guide

This guide explains the comprehensive visualization features available in the Enhanced Toxicity Evaluation Pipeline.

## 🎨 Overview

The visualization system provides interactive plots, statistical analysis, and comprehensive WandB integration for analyzing toxicity evaluation results. It's inspired by the original interactive analysis script but fully integrated into the evaluation pipeline.

## 🚀 Quick Start

### 1. Running Evaluation with Visualizations

```bash
python run_evaluation.py \
  models=pythia_70m_detox_all_epochs \
  dataset=custom_dataset_500 \
  output=comprehensive_wandb \
  visualization=comprehensive \
  generation.max_new_tokens=20 \
  generation.temperature=0.3 \
  generation.do_sample=false \
  experiment.name="pythia_70m_detox_comprehensive" \
  logging.use_wandb=true \
  logging.wandb_project="pythia-detoxification-analysis"
```

### 2. Analyzing Existing Results

```bash
python analyze_results.py path/to/your/results.csv \
  --wandb-project "my-analysis" \
  --experiment-name "custom_analysis"
```

## 📊 Visualization Types

### 1. Interactive Scatter Plots
- **Purpose**: Compare base model vs fine-tuned models across different classifiers
- **Features**: 
  - Interactive model selection via legend
  - Hover details with improvement metrics
  - Diagonal line showing "no change" baseline
  - Color-coded by model performance

### 2. Delta/Improvement Distributions
- **Purpose**: Analyze the distribution of improvements across prompts
- **Features**:
  - Histogram overlays for multiple models
  - Statistical summaries (mean, positive/negative rates)
  - Zero line indicating no change
  - Interactive hover details

### 3. Training Progression Plots
- **Purpose**: Track model improvement over training epochs
- **Features**:
  - Line plots with optional error bars
  - Multiple classifier comparison
  - Interactive epoch selection
  - Confidence intervals

### 4. Model Comparison Heatmaps
- **Purpose**: Compare models across all metrics simultaneously
- **Features**:
  - Color-coded performance matrix
  - Overall ranking visualization
  - Interactive hover details
  - Exportable comparison tables

### 5. Individual Prompt Tracking
- **Purpose**: Analyze how individual prompts change across training
- **Features**:
  - Trajectory plots for individual prompts
  - Color-coded by improvement (green=improved, red=regressed)
  - Average trajectory overlay
  - Interactive prompt selection

### 6. Statistical Analysis
- **Purpose**: Statistical significance testing and confidence intervals
- **Features**:
  - 95% confidence intervals
  - Statistical significance testing
  - Comprehensive metrics tables
  - Multiple comparison corrections

### 7. Text Type Analysis
- **Purpose**: Compare toxicity across different text types (prompt, output, full text)
- **Features**:
  - Violin plots for distribution comparison
  - Side-by-side model comparison
  - Statistical summaries

### 8. Advanced Interactive Dashboard
- **Purpose**: Multi-dimensional analysis with custom controls
- **Features**:
  - 3D scatter plots
  - Animated progression plots
  - Custom parameter selection
  - Advanced filtering options

## 🔧 Configuration

### Visualization Configuration File

The visualization behavior is controlled by `configs/visualization/comprehensive.yaml`:

```yaml
visualization:
  enabled: true
  save_plots_locally: true
  save_plots_format: "png"
  plot_dpi: 300
  plot_height: 600
  plot_width: 800
  
  interactive:
    enabled: true
    use_plotly: true
    hover_mode: "closest"
    
  scatter_plots:
    enabled: true
    marker_size: 4
    marker_opacity: 0.6
    show_diagonal_line: true
    
  # ... more configuration options
```

### Key Configuration Options

- **`enabled`**: Enable/disable visualization generation
- **`save_plots_locally`**: Save plots to local filesystem
- **`save_plots_format`**: Output format (png, pdf, svg, html)
- **`plot_dpi`**: Resolution for static images
- **`interactive.enabled`**: Enable interactive plotly plots
- **`wandb.log_plots`**: Log plots to WandB
- **`wandb.log_complete_csv`**: Log full CSV as WandB artifact

## 📈 WandB Integration

### Automatic Logging

The visualization system automatically logs to WandB:

1. **Complete CSV**: Full results as artifact
2. **Interactive Plots**: All plots as Plotly objects
3. **Tables**: Statistical summaries and comparison tables
4. **Metrics**: Comprehensive performance metrics

### WandB Artifacts

- `complete_results_{experiment_name}`: Full CSV dataset
- `complete_results_preview`: Sample table for quick preview

### WandB Plots

- `scatter_{classifier}_interactive`: Interactive scatter plots
- `delta_{classifier}_interactive`: Improvement distributions
- `progression_main_classifiers`: Training progression
- `model_comparison_heatmap`: Model comparison matrix
- `individual_prompt_trajectories`: Prompt tracking
- `statistical_significance`: Statistical analysis
- `text_type_comparison`: Text type analysis
- `3d_multi_classifier_analysis`: 3D analysis

## 🛠️ Usage Examples

### Example 1: Basic Evaluation with Visualizations

```python
# This will automatically create all visualizations
python run_evaluation.py \
  models=pythia_70m_detox_all_epochs \
  dataset=custom_dataset_500 \
  output=comprehensive_wandb \
  visualization=comprehensive
```

### Example 2: Custom Analysis of Existing Data

```python
# Analyze existing CSV file
python analyze_results.py results/experiment_123/toxicity_evaluation_results.csv \
  --wandb-project "custom-analysis" \
  --experiment-name "detailed_analysis" \
  --output-dir "custom_analysis_output"
```

### Example 3: Disable WandB for Local Analysis

```python
# Run analysis without WandB
python analyze_results.py results.csv --no-wandb
```

### Example 4: Custom Visualization Configuration

```yaml
# Create custom config
visualization:
  scatter_plots:
    marker_size: 6
    marker_opacity: 0.8
  delta_plots:
    histogram_bins: 50
  prompt_tracking:
    max_prompts_to_show: 100
```

## 📊 Output Structure

```
results/
├── experiment_name/
│   ├── plots/
│   │   ├── complete_results.csv          # Full results
│   │   ├── scatter_roberta_interactive.html
│   │   ├── delta_roberta_interactive.html
│   │   ├── progression_main_classifiers.html
│   │   └── ...
│   ├── analysis_metrics.json             # Calculated metrics
│   └── evaluation_summary.json           # Original summary
```

## 🔍 Key Features

### 1. Full CSV Storage in WandB
- Complete results logged as artifact
- Easy download and sharing
- Version control for results

### 2. Interactive Plots
- Plotly-based interactive visualizations
- Hover details with comprehensive information
- Legend-based model/classifier selection
- Zoom, pan, and export capabilities

### 3. Statistical Rigor
- Confidence intervals
- Significance testing
- Multiple comparison corrections
- Comprehensive statistical summaries

### 4. Individual Prompt Analysis
- Track how each prompt changes during training
- Identify problematic or successful prompts
- Understand model behavior at granular level

### 5. Multi-dimensional Analysis
- 3D visualizations for complex relationships
- Heatmaps for comprehensive comparisons
- Animated plots for temporal analysis

## 🎯 Best Practices

### 1. For Large Datasets
- Use subsampling for interactive plots
- Enable performance optimizations
- Consider batch processing for very large files

### 2. For WandB Integration
- Set appropriate project names
- Use descriptive experiment names
- Organize runs with tags

### 3. For Analysis
- Start with overview plots
- Drill down into specific areas of interest
- Use statistical analysis for validation

### 4. For Customization
- Modify color schemes for consistency
- Adjust plot sizes for your use case
- Configure export formats as needed

## 🐛 Troubleshooting

### Common Issues

1. **WandB Connection Issues**
   ```bash
   # Check WandB login
   wandb login
   
   # Or disable WandB
   --no-wandb
   ```

2. **Missing Dependencies**
   ```bash
   pip install plotly seaborn kaleido
   ```

3. **Large File Issues**
   ```yaml
   visualization:
     performance:
       max_data_points: 5000
       subsample_large_datasets: true
   ```

4. **Memory Issues**
   ```yaml
   visualization:
     prompt_tracking:
       max_prompts_to_show: 25  # Reduce from 50
   ```

## 📚 Advanced Usage

### Custom Visualization Functions

You can extend the visualization system by adding custom functions to `VisualizationManager`:

```python
def create_custom_analysis(self, df: pd.DataFrame):
    """Create custom analysis plots."""
    # Your custom visualization code here
    fig = go.Figure()
    # ... create plot
    self.log_plot_to_wandb(fig, "custom_analysis", "Custom analysis")
```

### Integration with Other Tools

The visualization system can be integrated with other analysis tools:

```python
from core.visualization_manager import VisualizationManager

# Create custom analysis
viz_manager = VisualizationManager(config, output_dir)
viz_manager.create_comprehensive_visualizations(df, metrics)
```

## 🤝 Contributing

To add new visualization types:

1. Add method to `VisualizationManager`
2. Update configuration schema
3. Add to `create_comprehensive_visualizations`
4. Update documentation

## 📄 License

This visualization system is part of the Enhanced Toxicity Evaluation Pipeline and follows the same license terms. 