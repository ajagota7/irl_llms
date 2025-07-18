#!/usr/bin/env python3
"""
Test script for the visualization system.
Creates sample data and tests the visualization functionality.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the current directory to the path for imports
sys.path.append(str(Path(__file__).parent))

from core.visualization_manager import VisualizationManager


def create_sample_data(n_prompts: int = 100) -> pd.DataFrame:
    """Create sample data for testing visualization."""
    print("📊 Creating sample data...")
    
    # Create sample prompts
    prompts = [f"This is sample prompt {i} with some content." for i in range(n_prompts)]
    
    # Create sample outputs for different models
    models = ['base', 'epoch_20', 'epoch_40', 'epoch_60', 'epoch_80', 'epoch_100']
    classifiers = ['roberta', 'dynabench']
    
    data = {'prompt': prompts}
    
    # Generate sample toxicity scores
    np.random.seed(42)
    
    for model in models:
        # Add output text
        data[f'output_{model}'] = [f"Sample output for {model} - prompt {i}" for i in range(n_prompts)]
        
        # Add full text
        data[f'full_{model}'] = [f"{prompt} {data[f'output_{model}'][i]}" for i, prompt in enumerate(prompts)]
        
        # Add toxicity scores for each classifier
        for classifier in classifiers:
            if model == 'base':
                # Base model has higher toxicity
                base_toxicity = np.random.beta(2, 1, n_prompts) * 0.8 + 0.2
                data[f'output_{model}_{classifier}_score'] = base_toxicity
                data[f'full_{model}_{classifier}_score'] = base_toxicity * 0.9
            else:
                # Fine-tuned models have lower toxicity (improvement)
                epoch_num = int(model.split('_')[1])
                improvement_factor = 1 - (epoch_num / 100) * 0.6  # Up to 60% improvement
                noise = np.random.normal(0, 0.1, n_prompts)
                
                model_toxicity = base_toxicity * improvement_factor + noise
                model_toxicity = np.clip(model_toxicity, 0, 1)
                
                data[f'output_{model}_{classifier}_score'] = model_toxicity
                data[f'full_{model}_{classifier}_score'] = model_toxicity * 0.9
    
    # Add prompt toxicity scores
    for classifier in classifiers:
        data[f'prompt_{classifier}_score'] = np.random.beta(1, 3, n_prompts) * 0.3
    
    # Create delta columns
    for model in models[1:]:  # Skip base model
        for classifier in classifiers:
            base_col = f'output_base_{classifier}_score'
            model_col = f'output_{model}_{classifier}_score'
            delta_col = f'delta_{model}_vs_base_{classifier}_score'
            data[delta_col] = data[base_col] - data[model_col]
    
    df = pd.DataFrame(data)
    print(f"✅ Created sample data with {len(df)} rows and {len(df.columns)} columns")
    return df


def create_test_config() -> dict:
    """Create test configuration."""
    config = {
        "experiment": {
            "name": "test_visualization",
            "seed": 42,
            "device": "auto"
        },
        "logging": {
            "use_wandb": False,  # Disable WandB for testing
            "wandb_project": "test-project",
            "wandb_entity": None,
            "log_level": "INFO",
            "save_logs": True
        },
        "visualization": {
            "enabled": True,
            "save_plots_locally": True,
            "save_plots_format": "html",  # Use HTML for testing
            "plot_dpi": 300,
            "plot_height": 600,
            "plot_width": 800,
            "interactive": {
                "enabled": True,
                "use_plotly": True,
                "hover_mode": "closest",
                "show_legend": True,
                "legend_position": "top-right"
            },
            "scatter_plots": {
                "enabled": True,
                "marker_size": 4,
                "marker_opacity": 0.6,
                "show_diagonal_line": True,
                "diagonal_line_color": "red",
                "diagonal_line_style": "dash"
            },
            "delta_plots": {
                "enabled": True,
                "histogram_bins": 40,
                "overlay_mode": True,
                "show_zero_line": True,
                "zero_line_color": "red",
                "zero_line_style": "dash"
            },
            "progression_plots": {
                "enabled": True,
                "show_error_bars": False,
                "error_bar_type": "data",
                "line_width": 3,
                "marker_size": 8,
                "show_zero_line": True
            },
            "comparison_plots": {
                "enabled": True,
                "heatmap_colorscale": "RdYlGn",
                "ranking_plot_type": "bar",
                "show_confidence_intervals": True
            },
            "prompt_tracking": {
                "enabled": True,
                "max_prompts_to_show": 20,  # Reduced for testing
                "trajectory_line_width": 1,
                "trajectory_opacity": 0.7,
                "show_average_trajectory": True,
                "average_line_width": 4,
                "average_line_color": "blue"
            },
            "statistical_analysis": {
                "enabled": True,
                "confidence_level": 0.95,
                "significance_test": "t_test",
                "show_p_values": True,
                "multiple_comparison_correction": "bonferroni"
            },
            "text_type_analysis": {
                "enabled": True,
                "violin_plot_mode": "overlay",
                "show_distribution_stats": True,
                "compare_prompt_output_full": True
            },
            "advanced_dashboard": {
                "enabled": True,
                "enable_3d_plots": True,
                "enable_animations": True,
                "animation_duration": 1000,
                "enable_custom_controls": True
            },
            "wandb": {
                "log_plots": False,  # Disable for testing
                "log_tables": False,
                "log_artifacts": False,
                "log_complete_csv": False,
                "csv_preview_rows": 1000,
                "artifact_name_pattern": "complete_results_{experiment_name}"
            }
        }
    }
    
    return config


def calculate_test_metrics(df: pd.DataFrame) -> dict:
    """Calculate basic metrics for testing."""
    print("📊 Calculating test metrics...")
    
    metrics = {
        "total_prompts": len(df),
        "total_columns": len(df.columns),
        "models_evaluated": len([col for col in df.columns if col.startswith("output_") and not col.endswith("_score")]),
        "classifiers_used": len([col for col in df.columns if col.endswith("_score") and "base" in col])
    }
    
    # Calculate model performance metrics
    model_metrics = {}
    model_cols = [col for col in df.columns if col.startswith("output_") and not col.endswith("_score")]
    
    for col in model_cols:
        model_name = col.replace("output_", "")
        score_cols = [c for c in df.columns if c.startswith(f"output_{model_name}_") and c.endswith("_score")]
        
        for score_col in score_cols:
            classifier_name = score_col.replace(f"output_{model_name}_", "").replace("_score", "")
            scores = df[score_col].dropna()
            
            if len(scores) > 0:
                model_metrics[f"{model_name}_{classifier_name}"] = {
                    "mean": scores.mean(),
                    "std": scores.std(),
                    "median": scores.median(),
                    "min": scores.min(),
                    "max": scores.max(),
                    "count": len(scores)
                }
    
    metrics["model_metrics"] = model_metrics
    
    # Calculate comparison metrics
    comparison_metrics = {}
    delta_cols = [col for col in df.columns if col.startswith("delta_")]
    
    for col in delta_cols:
        delta_data = df[col].dropna()
        
        if len(delta_data) > 0:
            comparison_metrics[col] = {
                "mean_improvement": delta_data.mean(),
                "std_improvement": delta_data.std(),
                "positive_count": (delta_data > 0.01).sum(),
                "negative_count": (delta_data < -0.01).sum(),
                "total_count": len(delta_data),
                "improved_rate": (delta_data > 0.01).mean()
            }
    
    metrics["comparison_metrics"] = comparison_metrics
    
    print("✅ Test metrics calculated")
    return metrics


def main():
    """Main test function."""
    print("🧪 Testing Visualization System")
    print("=" * 50)
    
    try:
        # Create test configuration
        config = create_test_config()
        
        # Create output directory
        output_dir = Path("test_visualization_output")
        output_dir.mkdir(exist_ok=True)
        
        # Create sample data
        df = create_sample_data(n_prompts=50)  # Use smaller dataset for testing
        
        # Calculate metrics
        metrics = calculate_test_metrics(df)
        
        # Save sample data
        csv_path = output_dir / "test_data.csv"
        df.to_csv(csv_path, index=False)
        print(f"💾 Saved test data to {csv_path}")
        
        # Initialize visualization manager
        print("🎨 Initializing visualization manager...")
        viz_manager = VisualizationManager(config, output_dir)
        
        # Create comprehensive visualizations
        print("📈 Creating visualizations...")
        viz_manager.create_comprehensive_visualizations(df, metrics)
        
        # Check output files
        plots_dir = output_dir / "plots"
        if plots_dir.exists():
            plot_files = list(plots_dir.glob("*.html"))
            print(f"✅ Created {len(plot_files)} HTML plot files:")
            for plot_file in plot_files:
                print(f"   - {plot_file.name}")
        
        # Cleanup
        viz_manager.cleanup()
        
        print("\n🎉 Visualization test completed successfully!")
        print(f"📁 Check output directory: {output_dir}")
        print("🌐 Open HTML files in your browser to view interactive plots")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 