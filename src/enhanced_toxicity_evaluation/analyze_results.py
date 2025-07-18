#!/usr/bin/env python3
"""
Standalone script for analyzing existing toxicity evaluation results and creating visualizations.
This script can be used to analyze CSV files from previous runs and create comprehensive visualizations.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import pandas as pd

# Add the current directory to the path for imports
sys.path.append(str(Path(__file__).parent))

from core.visualization_manager import VisualizationManager

logger = logging.getLogger(__name__)


def create_analysis_config(csv_path: str, output_dir: str = None, wandb_project: str = None, 
                          wandb_entity: str = None, experiment_name: str = None) -> DictConfig:
    """Create a configuration for analysis."""
    
    if output_dir is None:
        output_dir = f"analysis_results_{Path(csv_path).stem}"
    
    if experiment_name is None:
        experiment_name = f"analysis_{Path(csv_path).stem}"
    
    config = {
        "experiment": {
            "name": experiment_name,
            "seed": 42,
            "device": "auto"
        },
        "logging": {
            "use_wandb": True,
            "wandb_project": wandb_project or "toxicity-analysis",
            "wandb_entity": wandb_entity,
            "log_level": "INFO",
            "save_logs": True
        },
        "visualization": {
            "enabled": True,
            "save_plots_locally": True,
            "save_plots_format": "png",
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
                "max_prompts_to_show": 50,
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
                "log_plots": True,
                "log_tables": True,
                "log_artifacts": True,
                "log_complete_csv": True,
                "csv_preview_rows": 1000,
                "artifact_name_pattern": "complete_results_{experiment_name}"
            }
        },
        "analysis": {
            "csv_path": csv_path,
            "output_dir": output_dir,
            "create_delta_columns": True,
            "baseline_model": "base"
        }
    }
    
    return OmegaConf.create(config)


def load_and_prepare_data(csv_path: str, config: DictConfig) -> pd.DataFrame:
    """Load and prepare the CSV data for analysis."""
    logger.info(f"📊 Loading data from {csv_path}")
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    logger.info(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # Create delta columns if they don't exist
    if config.analysis.create_delta_columns:
        df = create_delta_columns(df, config.analysis.baseline_model)
    
    return df


def create_delta_columns(df: pd.DataFrame, baseline_model: str = "base") -> pd.DataFrame:
    """Create delta columns for comparison with baseline model."""
    logger.info(f"📊 Creating delta columns with baseline model: {baseline_model}")
    
    # Find output columns for baseline model
    baseline_output_cols = [col for col in df.columns if col.startswith(f"output_{baseline_model}_") and col.endswith("_score")]
    
    for col in baseline_output_cols:
        classifier_name = col.replace(f"output_{baseline_model}_", "").replace("_score", "")
        
        # Find corresponding columns for other models
        for model_col in df.columns:
            if model_col.startswith("output_") and model_col.endswith(f"_{classifier_name}_score") and model_col != col:
                model_name = model_col.replace("output_", "").replace(f"_{classifier_name}_score", "")
                
                if model_name != baseline_model:
                    delta_col = f"delta_{model_name}_vs_{baseline_model}_{classifier_name}_score"
                    df[delta_col] = df[col] - df[model_col]
    
    logger.info(f"✅ Created delta columns for comparison")
    return df


def calculate_basic_metrics(df: pd.DataFrame) -> dict:
    """Calculate basic metrics from the data."""
    logger.info("📊 Calculating basic metrics...")
    
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
    
    logger.info("✅ Basic metrics calculated")
    return metrics


def main():
    """Main function for analyzing results."""
    parser = argparse.ArgumentParser(description="Analyze toxicity evaluation results and create visualizations")
    parser.add_argument("csv_path", help="Path to the CSV file with evaluation results")
    parser.add_argument("--output-dir", help="Output directory for analysis results")
    parser.add_argument("--wandb-project", help="WandB project name")
    parser.add_argument("--wandb-entity", help="WandB entity/username")
    parser.add_argument("--experiment-name", help="Experiment name for WandB")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_path):
        logger.error(f"CSV file not found: {args.csv_path}")
        sys.exit(1)
    
    try:
        # Create configuration
        config = create_analysis_config(
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            experiment_name=args.experiment_name
        )
        
        # Disable WandB if requested
        if args.no_wandb:
            config.logging.use_wandb = False
        
        # Print configuration
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(config))
        
        # Load and prepare data
        df = load_and_prepare_data(args.csv_path, config)
        
        # Calculate metrics
        metrics = calculate_basic_metrics(df)
        
        # Create output directory
        output_dir = Path(config.analysis.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualization manager
        viz_manager = VisualizationManager(config, output_dir)
        
        # Create comprehensive visualizations
        logger.info("🎨 Creating comprehensive visualizations...")
        viz_manager.create_comprehensive_visualizations(df, metrics)
        
        # Save metrics to file
        metrics_path = output_dir / "analysis_metrics.json"
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"✅ Analysis completed successfully!")
        logger.info(f"📁 Results saved to: {output_dir}")
        logger.info(f"📊 Metrics saved to: {metrics_path}")
        
        if viz_manager.wandb_run:
            logger.info(f"🔗 WandB run: {viz_manager.wandb_run.get_url()}")
        
        # Cleanup
        viz_manager.cleanup()
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main() 