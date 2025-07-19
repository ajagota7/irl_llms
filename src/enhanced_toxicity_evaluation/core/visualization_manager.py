"""
Comprehensive visualization manager for the Enhanced Toxicity Evaluation Pipeline.
Provides interactive plots, statistical analysis, and WandB integration.
"""

import sys
import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import wandb
from typing import Dict, List, Tuple, Any, Optional
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class VisualizationManager:
    """Manages all visualization and analysis functionality for toxicity evaluation."""
    
    def __init__(self, config: Dict, output_dir: Path):
        """Initialize the visualization manager."""
        self.config = config
        self.output_dir = output_dir
        self.plots_dir = output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Color schemes
        self.model_colors = {
            'base': '#34495e',
            'epoch_20': '#3498db',
            'epoch_40': '#2ecc71', 
            'epoch_60': '#f39c12',
            'epoch_80': '#e74c3c',
            'epoch_100': '#9b59b6'
        }
        
        self.category_colors = {
            'toxic': '#e74c3c',
            'severe_toxic': '#c0392b',
            'obscene': '#f39c12',
            'threat': '#e67e22',
            'insult': '#9b59b6',
            'identity_hate': '#8e44ad'
        }
        
        self.classifier_colors = {
            'roberta': '#3498db',
            'dynabench': '#2ecc71'
        }
        
        # Extract configuration
        self.wandb_config = config.get("logging", {})
        self.visualization_config = config.get("visualization", {})
        self.experiment_config = config.get("experiment", {})
        
        # Initialize WandB if enabled
        self.wandb_run = None
        if self.wandb_config.get("use_wandb", True):
            self._setup_wandb()
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        try:
            # Convert config to dict for WandB
            if hasattr(self.config, 'to_container'):
                config_dict = self.config.to_container(resolve=True)
            elif isinstance(self.config, dict):
                config_dict = self.config
            else:
                config_dict = {}
            
            self.wandb_run = wandb.init(
                project=self.wandb_config.get("wandb_project", "toxicity-evaluation"),
                entity=self.wandb_config.get("wandb_entity"),
                name=self.experiment_config.get("name"),
                config=config_dict,
                tags=self.wandb_config.get("wandb_tags", [])
            )
            logger.info(f"WandB initialized: {self.wandb_run.get_url()}")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
            self.wandb_run = None
    
    def log_dataframe_as_table(self, df: pd.DataFrame, table_name: str, description: str = ""):
        """Log a DataFrame as a WandB table."""
        if self.wandb_run:
            wandb.log({
                table_name: wandb.Table(dataframe=df)
            })
    
    def log_plot_to_wandb(self, fig, plot_name: str, description: str = "", step: Optional[int] = None):
        """Log a plot to WandB."""
        if self.wandb_run:
            wandb.log({
                plot_name: wandb.Plotly(fig)
            }, step=step)
    
    def _detect_models_and_classifiers(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """Detect models, classifiers, and Toxic-BERT categories from column structure."""
        models = set()
        classifiers = set()
        toxic_bert_categories = set()
        
        # Known classifiers in your data
        known_classifiers = ['toxic_bert', 'roberta_toxicity', 'dynabench_hate']
        
        # Toxic-BERT categories
        toxic_bert_cat_list = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        # Debug: Print all columns to understand the structure
        logger.info(f"🔍 Analyzing DataFrame columns: {list(df.columns)}")
        
        # Look for results columns (standardized format)
        results_cols = [col for col in df.columns if col.endswith('_results')]
        logger.info(f"📊 Found results columns: {results_cols}")
        
        for col in results_cols:
            # Handle standardized patterns:
            # - prompt_{classifier}_results
            # - output_{classifier}_results
            # - full_text_{classifier}_results
            
            if col.startswith('prompt_'):
                # Format: prompt_{classifier}_results
                classifier = col.replace('prompt_', '').replace('_results', '')
                classifiers.add(classifier)
                
            elif col.startswith('output_'):
                # Format: output_{classifier}_results
                classifier = col.replace('output_', '').replace('_results', '')
                classifiers.add(classifier)
                        
            elif col.startswith('full_text_'):
                # Format: full_text_{classifier}_results
                classifier = col.replace('full_text_', '').replace('_results', '')
                classifiers.add(classifier)
        
        # Look for model column directly
        if 'model' in df.columns:
            unique_models = df['model'].unique()
            logger.info(f"📊 Found models in 'model' column: {unique_models}")
            models.update(unique_models)
        
        # Check for Toxic-BERT category columns
        for col in results_cols:
            for category in toxic_bert_cat_list:
                if category in col and 'toxic_bert' in col:
                    toxic_bert_categories.add(category)
        
        # Since column names are standardized, we need to infer models from the data structure
        # For now, use default models based on the configuration or file structure
        if not models:
            logger.info("📊 No models detected in columns, using default model set...")
            # Use the standard model progression
            models = {'base', 'detox_epoch_20', 'detox_epoch_40', 'detox_epoch_60', 'detox_epoch_80', 'detox_epoch_100'}
        
        # Ensure we have at least one classifier
        if not classifiers:
            logger.warning("⚠️ No classifiers detected, using default classifiers...")
            classifiers = {'toxic_bert', 'roberta_toxicity', 'dynabench_hate'}
        
        detected_models = sorted(list(models))
        detected_classifiers = sorted(list(classifiers))
        detected_categories = sorted(list(toxic_bert_categories))
        
        logger.info(f"✅ Detected models: {detected_models}")
        logger.info(f"✅ Detected classifiers: {detected_classifiers}")
        logger.info(f"✅ Detected Toxic-BERT categories: {detected_categories}")
        
        return detected_models, detected_classifiers, detected_categories

    def _create_delta_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create simple delta columns for basic comparisons."""
        logger.info("📊 Creating basic delta columns...")
        
        models, classifiers, toxic_bert_categories = self._detect_models_and_classifiers(df)
        logger.info(f"📋 Detected models: {models}")
        logger.info(f"🔍 Detected classifiers: {classifiers}")
        logger.info(f"🎯 Detected Toxic-BERT categories: {toxic_bert_categories}")
        
        df_copy = df.copy()
        
        # For now, just return the original DataFrame
        # The actual model comparisons will be done during visualization
        # when we have access to multiple model datasets
        
        logger.info("✅ Keeping original data structure - comparisons will be done during visualization")
        
        return df_copy
    
    def create_comprehensive_visualizations(self, df: pd.DataFrame, metrics: Dict[str, Any]):
        """Create comprehensive visualizations and log to WandB."""
        logger.info("🎨 Creating comprehensive visualizations...")
        
        # Create basic visualizations
        self._create_basic_plots(df)
        
        # Create advanced visualizations (if we have multiple models)
        if self._has_multiple_models(df):
            self._create_advanced_plots(df)
        
        # Log metrics to WandB
        self._log_comprehensive_metrics(df, metrics)
        
        logger.info("✅ All visualizations created and logged to WandB")
    
    def _has_multiple_models(self, df: pd.DataFrame) -> bool:
        """Check if we have multiple models for comparison."""
        if 'model' in df.columns:
            unique_models = df['model'].unique()
            return len(unique_models) > 1
        return False
    
    def _create_basic_plots(self, df: pd.DataFrame):
        """Create basic plots for single model data."""
        logger.info("📊 Creating basic plots...")
        
        models, classifiers, toxic_bert_categories = self._detect_models_and_classifiers(df)
        
        # Create distribution plots for each classifier
        for classifier in classifiers:
            for text_type in ['prompt', 'output', 'full_text']:
                col_name = f'{text_type}_{classifier}_results'
                
                if col_name in df.columns:
                    # Extract scores
                    scores = []
                    for result in df[col_name]:
                        if isinstance(result, dict):
                            if classifier == 'toxic_bert':
                                score = result.get('toxic', 0.0)
                            elif classifier == 'roberta_toxicity':
                                score = result.get('toxic', 0.0)
                            elif classifier == 'dynabench_hate':
                                score = result.get('hate', 0.0)
                            else:
                                score = next(iter(result.values()), 0.0)
                            scores.append(score)
                    
                    if scores:
                        # Create histogram
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=scores,
                            nbinsx=20,
                            name=f'{text_type} {classifier}',
                            opacity=0.7
                        ))
                        
                        fig.update_layout(
                            title=f'Distribution of {classifier} Scores - {text_type}',
                            xaxis_title='Toxicity Score',
                            yaxis_title='Count',
                            showlegend=True
                        )
                        
                        # Log to WandB
                        if self.wandb_run:
                            wandb.log({f"{text_type}_{classifier}_distribution": fig})
                        
                        # Save locally
                        plot_path = self.output_dir / f"{text_type}_{classifier}_distribution.html"
                        fig.write_html(str(plot_path))
                        logger.info(f"✅ Created {text_type}_{classifier}_distribution plot")
    
    def _create_advanced_plots(self, df: pd.DataFrame):
        """Create advanced plots for multi-model comparison."""
        logger.info("📊 Creating advanced multi-model plots...")
        
        # Group by model
        model_groups = df.groupby('model')
        models = list(model_groups.groups.keys())
        
        # Extract epoch numbers and create mapping
        model_epochs = {}
        for model_name in models:
            if model_name == "base":
                model_epochs[model_name] = 0
            elif "epoch" in model_name:
                try:
                    epoch_num = int(model_name.split("_")[-1])
                    model_epochs[model_name] = epoch_num
                except ValueError:
                    model_epochs[model_name] = 100
            else:
                model_epochs[model_name] = 100
        
        # Sort models by epoch
        sorted_models = sorted(model_epochs.items(), key=lambda x: x[1])
        model_names = [name for name, _ in sorted_models]
        epochs = [epoch for _, epoch in sorted_models]
        
        # Create toxicity reduction plots
        self._create_toxicity_reduction_plots(model_groups, model_names, epochs)
        
        # Create scatter plots
        self._create_scatter_plots(model_groups, model_names, epochs)
        
        # Create distribution plots
        self._create_improvement_distribution_plots(model_groups, model_names, epochs)
        
        # Create comprehensive dashboard
        self._create_comprehensive_dashboard(model_groups, model_names, epochs)
    
    def _create_toxicity_reduction_plots(self, model_groups, model_names: List[str], epochs: List[int]):
        """Create toxicity reduction plots across epochs."""
        logger.info("📊 Creating toxicity reduction plots...")
        
        classifiers = ["toxic_bert", "roberta_toxicity", "dynabench_hate"]
        text_types = ["output", "full_text"]
        
        for classifier in classifiers:
            for text_type in text_types:
                # Calculate mean scores for each model
                mean_scores = []
                for model_name in model_names:
                    model_df = model_groups.get_group(model_name)
                    col_name = f"{text_type}_{classifier}_results"
                    
                    if col_name in model_df.columns:
                        scores = []
                        for result in model_df[col_name]:
                            if isinstance(result, dict):
                                if classifier == "toxic_bert":
                                    score = result.get('toxic', 0.0)
                                elif classifier == "roberta_toxicity":
                                    score = result.get('toxic', 0.0)
                                elif classifier == "dynabench_hate":
                                    score = result.get('hate', 0.0)
                                else:
                                    score = next(iter(result.values()), 0.0)
                                scores.append(score)
                        
                        if scores:
                            mean_scores.append(np.mean(scores))
                        else:
                            mean_scores.append(0.0)
                    else:
                        mean_scores.append(0.0)
                
                # Create plot
                fig = go.Figure()
                
                # Add mean toxicity line
                fig.add_trace(go.Scatter(
                    x=epochs,
                    y=mean_scores,
                    mode='lines+markers',
                    name='Mean Toxicity',
                    line=dict(width=3),
                    marker=dict(size=8)
                ))
                
                # Add individual prompt scores as scatter
                all_scores = []
                all_epochs = []
                all_prompts = []
                
                for i, model_name in enumerate(model_names):
                    model_df = model_groups.get_group(model_name)
                    col_name = f"{text_type}_{classifier}_results"
                    
                    if col_name in model_df.columns:
                        for j, result in enumerate(model_df[col_name]):
                            if isinstance(result, dict):
                                if classifier == "toxic_bert":
                                    score = result.get('toxic', 0.0)
                                elif classifier == "roberta_toxicity":
                                    score = result.get('toxic', 0.0)
                                elif classifier == "dynabench_hate":
                                    score = result.get('hate', 0.0)
                                else:
                                    score = next(iter(result.values()), 0.0)
                                
                                all_scores.append(score)
                                all_epochs.append(epochs[i])
                                all_prompts.append(j)
                
                # Add scatter plot
                fig.add_trace(go.Scatter(
                    x=all_epochs,
                    y=all_scores,
                    mode='markers',
                    name='Individual Prompts',
                    marker=dict(
                        size=6,
                        color=all_prompts,
                        colorscale='viridis',
                        opacity=0.6,
                        showscale=True,
                        colorbar=dict(title="Prompt Index")
                    ),
                    hovertemplate='<b>Prompt %{marker.color}</b><br>Epoch: %{x}<br>Toxicity: %{y:.3f}<extra></extra>'
                ))
                
                # Add improvement annotation
                if len(mean_scores) > 1:
                    improvement = mean_scores[0] - mean_scores[-1]
                    improvement_pct = (improvement / mean_scores[0]) * 100 if mean_scores[0] > 0 else 0
                    
                    fig.add_annotation(
                        x=epochs[-1],
                        y=mean_scores[-1],
                        text=f'Total Reduction: {improvement:.3f} ({improvement_pct:.1f}%)',
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor="red",
                        ax=20,
                        ay=-30
                    )
                
                fig.update_layout(
                    title=f'Toxicity Reduction: {classifier} - {text_type}',
                    xaxis_title='Epoch',
                    yaxis_title='Toxicity Score',
                    showlegend=True
                )
                
                # Log to WandB
                if self.wandb_run:
                    wandb.log({f"toxicity_reduction_{classifier}_{text_type}": fig})
                
                # Save locally
                plot_path = self.output_dir / f"toxicity_reduction_{classifier}_{text_type}.html"
                fig.write_html(str(plot_path))
                logger.info(f"✅ Created toxicity reduction plot for {classifier} - {text_type}")
    
    def _create_scatter_plots(self, model_groups, model_names: List[str], epochs: List[int]):
        """Create scatter plots comparing base vs detoxified models."""
        logger.info("📊 Creating scatter plots...")
        
        classifiers = ["toxic_bert", "roberta_toxicity", "dynabench_hate"]
        text_types = ["output", "full_text"]
        
        # Get base model data
        base_model = model_names[0] if model_names else None
        if not base_model:
            return
        
        base_df = model_groups.get_group(base_model)
        
        for classifier in classifiers:
            for text_type in text_types:
                # Get base scores
                base_col = f"{text_type}_{classifier}_results"
                base_scores = []
                
                if base_col in base_df.columns:
                    for result in base_df[base_col]:
                        if isinstance(result, dict):
                            if classifier == "toxic_bert":
                                score = result.get('toxic', 0.0)
                            elif classifier == "roberta_toxicity":
                                score = result.get('toxic', 0.0)
                            elif classifier == "dynabench_hate":
                                score = result.get('hate', 0.0)
                            else:
                                score = next(iter(result.values()), 0.0)
                            base_scores.append(score)
                
                if not base_scores:
                    continue
                
                # Create scatter plot
                fig = go.Figure()
                
                # Add diagonal reference line
                max_score = max(max(base_scores), 1.0)
                fig.add_trace(go.Scatter(
                    x=[0, max_score],
                    y=[0, max_score],
                    mode='lines',
                    name='No Change',
                    line=dict(color='red', dash='dash'),
                    showlegend=True
                ))
                
                # Add detoxified models
                for model_name in model_names[1:]:  # Skip base model
                    model_df = model_groups.get_group(model_name)
                    detox_col = f"{text_type}_{classifier}_results"
                    detox_scores = []
                    
                    if detox_col in model_df.columns:
                        for result in model_df[detox_col]:
                            if isinstance(result, dict):
                                if classifier == "toxic_bert":
                                    score = result.get('toxic', 0.0)
                                elif classifier == "roberta_toxicity":
                                    score = result.get('toxic', 0.0)
                                elif classifier == "dynabench_hate":
                                    score = result.get('hate', 0.0)
                                else:
                                    score = next(iter(result.values()), 0.0)
                                detox_scores.append(score)
                    
                    if detox_scores and len(detox_scores) == len(base_scores):
                        fig.add_trace(go.Scatter(
                            x=base_scores,
                            y=detox_scores,
                            mode='markers',
                            name=f'{model_name} (Epoch {model_name.split("_")[-1] if "epoch" in model_name else "N/A"})',
                            marker=dict(size=8, opacity=0.7),
                            hovertemplate='<b>Prompt %{text}</b><br>Base: %{x:.3f}<br>Detoxified: %{y:.3f}<extra></extra>',
                            text=[f"{i+1}" for i in range(len(base_scores))]
                        ))
                
                fig.update_layout(
                    title=f'Base vs Detoxified Toxicity: {classifier} - {text_type}',
                    xaxis_title='Base Toxicity Score',
                    yaxis_title='Detoxified Toxicity Score',
                    showlegend=True
                )
                
                # Log to WandB
                if self.wandb_run:
                    wandb.log({f"scatter_{classifier}_{text_type}": fig})
                
                # Save locally
                plot_path = self.output_dir / f"scatter_{classifier}_{text_type}.html"
                fig.write_html(str(plot_path))
                logger.info(f"✅ Created scatter plot for {classifier} - {text_type}")
    
    def _create_improvement_distribution_plots(self, model_groups, model_names: List[str], epochs: List[int]):
        """Create distribution plots of improvements."""
        logger.info("📊 Creating improvement distribution plots...")
        
        classifiers = ["toxic_bert", "roberta_toxicity", "dynabench_hate"]
        text_types = ["output", "full_text"]
        
        # Get base model data
        base_model = model_names[0] if model_names else None
        if not base_model:
            return
        
        base_df = model_groups.get_group(base_model)
        
        for classifier in classifiers:
            for text_type in text_types:
                # Get base scores
                base_col = f"{text_type}_{classifier}_results"
                base_scores = []
                
                if base_col in base_df.columns:
                    for result in base_df[base_col]:
                        if isinstance(result, dict):
                            if classifier == "toxic_bert":
                                score = result.get('toxic', 0.0)
                            elif classifier == "roberta_toxicity":
                                score = result.get('toxic', 0.0)
                            elif classifier == "dynabench_hate":
                                score = result.get('hate', 0.0)
                            else:
                                score = next(iter(result.values()), 0.0)
                            base_scores.append(score)
                
                if not base_scores:
                    continue
                
                # Create distribution plot
                fig = go.Figure()
                
                # Calculate improvements for each detoxified model
                for model_name in model_names[1:]:  # Skip base model
                    model_df = model_groups.get_group(model_name)
                    detox_col = f"{text_type}_{classifier}_results"
                    improvements = []
                    
                    if detox_col in model_df.columns:
                        for i, result in enumerate(model_df[detox_col]):
                            if i < len(base_scores) and isinstance(result, dict):
                                if classifier == "toxic_bert":
                                    detox_score = result.get('toxic', 0.0)
                                elif classifier == "roberta_toxicity":
                                    detox_score = result.get('toxic', 0.0)
                                elif classifier == "dynabench_hate":
                                    detox_score = result.get('hate', 0.0)
                                else:
                                    detox_score = next(iter(result.values()), 0.0)
                                
                                improvement = base_scores[i] - detox_score
                                improvements.append(improvement)
                    
                    if improvements:
                        fig.add_trace(go.Histogram(
                            x=improvements,
                            name=f'{model_name} (Epoch {model_name.split("_")[-1] if "epoch" in model_name else "N/A"})',
                            opacity=0.7,
                            nbinsx=20
                        ))
                
                fig.update_layout(
                    title=f'Distribution of Toxicity Improvements: {classifier} - {text_type}',
                    xaxis_title='Improvement (Base - Detoxified)',
                    yaxis_title='Count',
                    barmode='overlay',
                    showlegend=True
                )
                
                # Log to WandB
                if self.wandb_run:
                    wandb.log({f"improvement_distribution_{classifier}_{text_type}": fig})
                
                # Save locally
                plot_path = self.output_dir / f"improvement_distribution_{classifier}_{text_type}.html"
                fig.write_html(str(plot_path))
                logger.info(f"✅ Created improvement distribution plot for {classifier} - {text_type}")
    
    def _create_comprehensive_dashboard(self, model_groups, model_names: List[str], epochs: List[int]):
        """Create a comprehensive dashboard with all metrics."""
        logger.info("📊 Creating comprehensive dashboard...")
        
        classifiers = ["toxic_bert", "roberta_toxicity", "dynabench_hate"]
        text_types = ["output", "full_text"]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Toxic-Bert Output', 'Roberta Output', 'Dynabench Output',
                'Toxic-Bert Full Text', 'Roberta Full Text', 'Dynabench Full Text'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add mean toxicity progression for each classifier and text type
        for i, classifier in enumerate(classifiers):
            for j, text_type in enumerate(text_types):
                row = j + 1
                col = i + 1
                
                mean_scores = []
                for model_name in model_names:
                    model_df = model_groups.get_group(model_name)
                    col_name = f"{text_type}_{classifier}_results"
                    
                    if col_name in model_df.columns:
                        scores = []
                        for result in model_df[col_name]:
                            if isinstance(result, dict):
                                if classifier == "toxic_bert":
                                    scores.append(result.get('toxic', 0.0))
                                elif classifier == "roberta_toxicity":
                                    scores.append(result.get('toxic', 0.0))
                                elif classifier == "dynabench_hate":
                                    scores.append(result.get('hate', 0.0))
                        
                        if scores:
                            mean_scores.append(np.mean(scores))
                        else:
                            mean_scores.append(0.0)
                    else:
                        mean_scores.append(0.0)
                
                if mean_scores:
                    fig.add_trace(
                        go.Scatter(
                            x=epochs,
                            y=mean_scores,
                            mode='lines+markers',
                            name=f'{classifier} {text_type}',
                            line=dict(width=3),
                            marker=dict(size=8),
                            showlegend=False
                        ),
                        row=row, col=col
                    )
                    
                    # Add improvement annotation
                    if len(mean_scores) > 1:
                        improvement = mean_scores[0] - mean_scores[-1]
                        improvement_pct = (improvement / mean_scores[0]) * 100 if mean_scores[0] > 0 else 0
                        fig.add_annotation(
                            x=epochs[-1],
                            y=mean_scores[-1],
                            text=f'{improvement_pct:.1f}%',
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor="red",
                            ax=20,
                            ay=-30,
                            row=row, col=col
                        )
        
        fig.update_layout(
            title='Comprehensive Toxicity Reduction Dashboard',
            height=800,
            showlegend=False
        )
        
        # Update axes labels
        for i in range(1, 3):
            for j in range(1, 4):
                fig.update_xaxes(title_text="Epoch", row=i, col=j)
                fig.update_yaxes(title_text="Mean Toxicity", row=i, col=j)
        
        # Log to WandB
        if self.wandb_run:
            wandb.log({"comprehensive_dashboard": fig})
        
        # Save locally
        dashboard_path = self.output_dir / "comprehensive_dashboard.html"
        fig.write_html(str(dashboard_path))
        logger.info("✅ Created comprehensive dashboard")
    
    def _log_full_csv_to_wandb(self, results_df: pd.DataFrame):
        """Log the complete CSV to WandB as an artifact."""
        if not self.wandb_run:
            return
        
        try:
            # Save CSV to plots directory
            csv_path = self.plots_dir / "complete_results.csv"
            results_df.to_csv(csv_path, index=False)
            
            # Create WandB artifact
            artifact = wandb.Artifact(
                name=f"complete_results_{self.experiment_config.get('name', 'default')}",
                type="dataset",
                description="Complete toxicity evaluation results"
            )
            artifact.add_file(str(csv_path))
            
            # Log the artifact
            wandb.log_artifact(artifact)
            
            # Also log as table (sample for preview)
            sample_df = results_df.head(1000)  # Log first 1000 rows as preview
            wandb.log({"complete_results_preview": wandb.Table(dataframe=sample_df)})
            
            logger.info(f"✅ Complete CSV logged to WandB: {csv_path}")
            
        except Exception as e:
            logger.warning(f"Failed to log complete CSV to WandB: {e}")
    
    def create_interactive_scatter_plots(self, df: pd.DataFrame):
        """Create interactive scatter plots and log to WandB."""
        logger.info("📈 Creating interactive scatter plots...")
        
        # Get model names from columns - handle your specific naming pattern
        model_cols = [col for col in df.columns if col.startswith('output_') and not col.endswith('_score')]
        models = [col.replace('output_', '') for col in model_cols]
        
        # Get classifier names from score columns
        classifier_cols = [col for col in df.columns if col.endswith('_score') and 'base' in col]
        classifiers = []
        for col in classifier_cols:
            # Handle different patterns: base_roberta_toxicity_score, base_dynabench_hate_score, etc.
            if col.startswith('base_'):
                classifier_name = col.replace('base_', '').replace('_score', '')
                classifiers.append(classifier_name)
        
        logger.info(f"Found models: {models}")
        logger.info(f"Found classifiers: {classifiers}")
        
        # 1. Main classifiers scatter plots with model selection
        for classifier in classifiers:
            fig = go.Figure()
            
            # Add traces for each model
            for model in models:
                # Handle your specific naming pattern
                if model == 'base':
                    base_col = f'base_{classifier}_score'
                else:
                    base_col = f'base_{classifier}_score'
                    model_col = f'{model}_{classifier}_score'
                
                if base_col in df.columns and model_col in df.columns:
                    # Remove NaN values
                    mask = ~(df[base_col].isna() | df[model_col].isna())
                    x_data = df[mask][base_col]
                    y_data = df[mask][model_col]
                    
                    if len(x_data) > 0:
                        # Calculate improvement metrics
                        improvement = (x_data - y_data).mean()
                        improved_count = (x_data > y_data).sum()
                        
                        fig.add_trace(go.Scatter(
                            x=x_data,
                            y=y_data,
                            mode='markers',
                            name=f'{model} (Δ={improvement:.4f})',
                            marker=dict(
                                color=self.model_colors.get(model, '#95a5a6'), 
                                size=4, 
                                opacity=0.6
                            ),
                            hovertemplate=f'<b>{model}</b><br>' +
                                        f'Base: %{{x:.4f}}<br>' +
                                        f'{model}: %{{y:.4f}}<br>' +
                                        f'Improvement: %{{text:.4f}}<br>' +
                                        '<extra></extra>',
                            text=[x_data.iloc[i] - y_data.iloc[i] for i in range(len(x_data))],
                            visible=True if model == models[-1] else 'legendonly'  # Show last model by default
                        ))
            
            # Add diagonal line
            max_val = 1.0  # Set reasonable max for toxicity scores
            fig.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                name='No Change',
                line=dict(color='red', dash='dash', width=2),
                hoverinfo='none',
                showlegend=True
            ))
            
            fig.update_layout(
                title=f'{classifier.upper()} Classifier: Base vs Fine-tuned Models (Interactive)',
                xaxis_title='Base Model Toxicity Score',
                yaxis_title='Fine-tuned Model Toxicity Score',
                xaxis=dict(range=[0, max_val]),
                yaxis=dict(range=[0, max_val]),
                hovermode='closest',
                height=600,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            # Add annotation
            fig.add_annotation(
                text="Points below diagonal = Improvement<br>Click legend to toggle models",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            )
            
            self.log_plot_to_wandb(fig, f"scatter_{classifier}_interactive", 
                                 f"Interactive scatter plot for {classifier} classifier")
    
    def create_interactive_delta_plots(self, df: pd.DataFrame):
        """Create interactive delta/improvement plots and log to WandB."""
        logger.info("📊 Creating interactive delta/improvement plots...")
        
        # Create delta columns if they don't exist
        df = self._create_delta_columns(df)
        
        # Get delta columns
        delta_cols = [col for col in df.columns if col.startswith('delta_')]
        
        # Group by classifier
        classifiers = set()
        for col in delta_cols:
            # Parse delta column names like: delta_detox_epoch_20_vs_base_roberta_toxicity_score
            parts = col.split('_')
            if len(parts) >= 6:  # delta_model_vs_base_classifier_score
                # Extract classifier name (everything after 'base_')
                base_index = parts.index('base') if 'base' in parts else -1
                if base_index != -1 and base_index + 1 < len(parts):
                    classifier = '_'.join(parts[base_index + 1:-1])  # Remove 'delta_' prefix and '_score' suffix
                    classifiers.add(classifier)
        
        logger.info(f"Found delta columns: {delta_cols}")
        logger.info(f"Found classifiers in delta plots: {classifiers}")
        
        # 1. Main classifiers improvement distributions
        for classifier in classifiers:
            fig = go.Figure()
            
            # Get all delta columns for this classifier
            classifier_delta_cols = [col for col in delta_cols if col.endswith(f'{classifier}_score')]
            
            for col in classifier_delta_cols:
                model_name = col.split('_')[1]  # Extract model name
                delta_data = df[col].dropna()
                
                if len(delta_data) > 0:
                    # Calculate statistics
                    mean_improvement = delta_data.mean()
                    positive_improvements = (delta_data > 0.01).sum()
                    negative_improvements = (delta_data < -0.01).sum()
                    
                    fig.add_trace(go.Histogram(
                        x=delta_data,
                        name=f'{model_name} (μ={mean_improvement:.4f})',
                        opacity=0.7,
                        marker_color=self.model_colors.get(model_name, '#95a5a6'),
                        nbinsx=40,
                        hovertemplate=f'<b>{model_name}</b><br>' +
                                    'Improvement: %{x:.4f}<br>' +
                                    'Count: %{y}<br>' +
                                    f'Positive: {positive_improvements}<br>' +
                                    f'Negative: {negative_improvements}<br>' +
                                    '<extra></extra>',
                        visible=True
                    ))
            
            # Add vertical line at x=0
            fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=2,
                         annotation_text="No Change")
            
            fig.update_layout(
                title=f'{classifier.upper()} Improvement Distribution (Interactive)',
                xaxis_title='Toxicity Reduction (positive = better)',
                yaxis_title='Frequency',
                barmode='overlay',
                height=600,
                showlegend=True,
                hovermode='x unified'
            )
            
            self.log_plot_to_wandb(fig, f"delta_{classifier}_interactive", 
                                 f"Interactive improvement distribution for {classifier}")
    
    def create_interactive_progression_plots(self, df: pd.DataFrame):
        """Create interactive progression plots and log to WandB."""
        logger.info("📈 Creating interactive progression plots...")
        
        # Extract epoch numbers from model names - more robust parsing
        epoch_models = {}
        for col in df.columns:
            if col.startswith('output_') and not col.endswith('_score'):
                model_name = col.replace('output_', '')
                
                # Try different patterns for epoch extraction
                epoch_num = None
                
                # Pattern 1: epoch_20, epoch_40, etc.
                if 'epoch_' in model_name:
                    try:
                        epoch_num = int(model_name.split('_')[1])
                    except (ValueError, IndexError):
                        pass
                
                # Pattern 2: check_20, check_40, etc.
                elif 'check_' in model_name:
                    try:
                        epoch_num = int(model_name.split('_')[1])
                    except (ValueError, IndexError):
                        pass
                
                # Pattern 3: step_20, step_40, etc.
                elif 'step_' in model_name:
                    try:
                        epoch_num = int(model_name.split('_')[1])
                    except (ValueError, IndexError):
                        pass
                
                # Pattern 4: any number at the end
                else:
                    # Try to extract any number from the model name
                    import re
                    numbers = re.findall(r'\d+', model_name)
                    if numbers:
                        try:
                            epoch_num = int(numbers[-1])  # Take the last number
                        except (ValueError, IndexError):
                            pass
                
                if epoch_num is not None:
                    epoch_models[epoch_num] = model_name
        
        if not epoch_models:
            logger.warning("No epoch-based models found for progression plots")
            logger.info(f"Available model columns: {[col for col in df.columns if col.startswith('output_') and not col.endswith('_score')]}")
            return
        
        epochs = sorted(epoch_models.keys())
        logger.info(f"Found epoch models: {epoch_models}")
        
        # 1. Main classifiers progression
        fig = go.Figure()
        
        # Get classifier names
        classifier_cols = [col for col in df.columns if col.endswith('_score') and 'base' in col]
        classifiers = [col.replace('output_base_', '').replace('_score', '') for col in classifier_cols]
        
        for classifier in classifiers:
            improvements = []
            std_devs = []
            positive_counts = []
            
            for epoch in epochs:
                model = epoch_models[epoch]
                delta_col = f'delta_{model}_vs_base_{classifier}_score'
                
                if delta_col in df.columns:
                    delta_data = df[delta_col].dropna()
                    improvements.append(delta_data.mean())
                    std_devs.append(delta_data.std())
                    positive_counts.append((delta_data > 0.01).sum())
                else:
                    improvements.append(0)
                    std_devs.append(0)
                    positive_counts.append(0)
            
            fig.add_trace(go.Scatter(
                x=epochs,
                y=improvements,
                mode='lines+markers',
                name=f'{classifier.upper()}',
                line=dict(color=self.classifier_colors.get(classifier, '#95a5a6'), width=3),
                marker=dict(size=8),
                error_y=dict(type='data', array=std_devs, visible=False),
                hovertemplate=f'<b>{classifier.upper()}</b><br>' +
                            'Epoch: %{x}<br>' +
                            'Avg Improvement: %{y:.4f}<br>' +
                            'Std Dev: %{error_y.array:.4f}<br>' +
                            'Positive Cases: %{text}<br>' +
                            '<extra></extra>',
                text=positive_counts
            ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2,
                     annotation_text="No Change")
        
        fig.update_layout(
            title='Main Classifiers: Average Improvement Over Training (Interactive)',
            xaxis_title='Training Epoch',
            yaxis_title='Average Toxicity Reduction',
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Add buttons to toggle error bars
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"error_y.visible": False}],
                            label="Hide Error Bars",
                            method="restyle"
                        ),
                        dict(
                            args=[{"error_y.visible": True}],
                            label="Show Error Bars",
                            method="restyle"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                ),
            ]
        )
        
        self.log_plot_to_wandb(fig, "progression_main_classifiers", 
                             "Interactive progression plot for main classifiers")
    
    def create_model_comparison_plots(self, df: pd.DataFrame):
        """Create interactive model comparison plots."""
        logger.info("🏆 Creating model comparison plots...")
        
        # Get model names and create comparison data
        model_cols = [col for col in df.columns if col.startswith('output_') and not col.endswith('_score')]
        models = [col.replace('output_', '') for col in model_cols]
        
        # Get classifier names
        classifier_cols = [col for col in df.columns if col.endswith('_score') and 'base' in col]
        classifiers = [col.replace('output_base_', '').replace('_score', '') for col in classifier_cols]
        
        comparison_data = []
        
        for model in models:
            row_data = {'model': model}
            
            # Main classifiers
            for classifier in classifiers:
                delta_col = f'delta_{model}_vs_base_{classifier}_score'
                if delta_col in df.columns:
                    avg_improvement = df[delta_col].mean()
                    row_data[f'{classifier}_improvement'] = avg_improvement
            
            comparison_data.append(row_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create heatmap
        metrics = [col for col in comparison_df.columns if col != 'model']
        if metrics:
            z_data = comparison_df[metrics].values
            
            fig = go.Figure(data=go.Heatmap(
                z=z_data,
                x=metrics,
                y=comparison_df['model'],
                colorscale='RdYlGn',
                colorbar=dict(title="Improvement Score"),
                hovertemplate='Model: %{y}<br>Metric: %{x}<br>Improvement: %{z:.4f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Model Performance Heatmap: Improvement Across All Metrics',
                xaxis_title='Metrics',
                yaxis_title='Models',
                height=600,
                xaxis=dict(tickangle=45)
            )
            
            self.log_plot_to_wandb(fig, "model_comparison_heatmap", 
                                 "Heatmap comparing model performance across all metrics")
            
            # Log comparison table
            self.log_dataframe_as_table(comparison_df, "model_comparison_table", 
                                      "Detailed comparison of model improvements")
            
            # Overall ranking plot
            overall_scores = []
            for _, row in comparison_df.iterrows():
                metric_scores = [row[col] for col in metrics if not pd.isna(row[col])]
                overall_score = np.mean(metric_scores) if metric_scores else 0
                overall_scores.append(overall_score)
            
            comparison_df['overall_score'] = overall_scores
            comparison_df_sorted = comparison_df.sort_values('overall_score', ascending=False)
            
            fig = go.Figure(data=go.Bar(
                x=comparison_df_sorted['model'],
                y=comparison_df_sorted['overall_score'],
                marker_color=[self.model_colors.get(model, '#95a5a6') for model in comparison_df_sorted['model']],
                hovertemplate='Model: %{x}<br>Overall Score: %{y:.4f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Overall Model Ranking: Average Improvement Across All Metrics',
                xaxis_title='Models',
                yaxis_title='Average Improvement Score',
                height=500
            )
            
            self.log_plot_to_wandb(fig, "model_ranking", "Overall model ranking based on average improvement")
    
    def create_individual_prompt_tracking(self, df: pd.DataFrame):
        """Create interactive plots showing individual prompt changes across epochs."""
        logger.info("🔍 Creating individual prompt tracking plots...")
        
        # Extract epoch numbers from model names - more robust parsing
        epoch_models = {}
        for col in df.columns:
            if col.startswith('output_') and not col.endswith('_score'):
                model_name = col.replace('output_', '')
                
                # Try different patterns for epoch extraction
                epoch_num = None
                
                # Pattern 1: epoch_20, epoch_40, etc.
                if 'epoch_' in model_name:
                    try:
                        epoch_num = int(model_name.split('_')[1])
                    except (ValueError, IndexError):
                        pass
                
                # Pattern 2: check_20, check_40, etc.
                elif 'check_' in model_name:
                    try:
                        epoch_num = int(model_name.split('_')[1])
                    except (ValueError, IndexError):
                        pass
                
                # Pattern 3: step_20, step_40, etc.
                elif 'step_' in model_name:
                    try:
                        epoch_num = int(model_name.split('_')[1])
                    except (ValueError, IndexError):
                        pass
                
                # Pattern 4: any number at the end
                else:
                    # Try to extract any number from the model name
                    import re
                    numbers = re.findall(r'\d+', model_name)
                    if numbers:
                        try:
                            epoch_num = int(numbers[-1])  # Take the last number
                        except (ValueError, IndexError):
                            pass
                
                if epoch_num is not None:
                    epoch_models[epoch_num] = model_name
        
        if not epoch_models:
            logger.warning("No epoch-based models found for prompt tracking")
            logger.info(f"Available model columns: {[col for col in df.columns if col.startswith('output_') and not col.endswith('_score')]}")
            return
        
        epochs = sorted(epoch_models.keys())
        logger.info(f"Found epoch models for prompt tracking: {epoch_models}")
        classifier = 'roberta'  # Use RoBERTa for main analysis
        
        # 1. Prompt trajectory plot - shows how each prompt changes over epochs
        fig = go.Figure()
        
        # Sample a subset of prompts for readability
        n_prompts_to_show = min(50, len(df))
        prompt_indices = np.random.choice(len(df), n_prompts_to_show, replace=False)
        
        for i, idx in enumerate(prompt_indices):
            prompt_scores = []
            hover_texts = []
            
            # Get base score
            base_col = f'output_base_{classifier}_score'
            base_score = df.iloc[idx][base_col] if base_col in df.columns else 0
            
            for epoch in epochs:
                model = epoch_models[epoch]
                score_col = f'output_{model}_{classifier}_score'
                
                if score_col in df.columns:
                    score = df.iloc[idx][score_col]
                    prompt_scores.append(score)
                    
                    # Create hover text with prompt and output info
                    prompt_text = df.iloc[idx]['prompt'][:100] + "..." if len(df.iloc[idx]['prompt']) > 100 else df.iloc[idx]['prompt']
                    output_text = df.iloc[idx][f'output_{model}'][:50] + "..." if len(str(df.iloc[idx][f'output_{model}'])) > 50 else str(df.iloc[idx][f'output_{model}'])
                    
                    hover_texts.append(f"Prompt: {prompt_text}<br>Output: {output_text}<br>Score: {score:.4f}")
                else:
                    prompt_scores.append(None)
                    hover_texts.append("No data")
            
            # Only plot if we have valid data
            if any(s is not None for s in prompt_scores):
                # Color by improvement (green = improved, red = worse)
                final_improvement = base_score - (prompt_scores[-1] if prompt_scores[-1] is not None else base_score)
                line_color = 'green' if final_improvement > 0.05 else 'red' if final_improvement < -0.05 else 'gray'
                
                fig.add_trace(go.Scatter(
                    x=epochs,
                    y=prompt_scores,
                    mode='lines+markers',
                    name=f'Prompt {i+1}',
                    line=dict(color=line_color, width=1),
                    marker=dict(size=4),
                    opacity=0.7,
                    hovertemplate='<b>Prompt %{name}</b><br>' +
                                'Epoch: %{x}<br>' +
                                '%{text}<extra></extra>',
                    text=hover_texts,
                    showlegend=False
                ))
        
        # Add average trajectory
        avg_scores = []
        for epoch in epochs:
            model = epoch_models[epoch]
            score_col = f'output_{model}_{classifier}_score'
            if score_col in df.columns:
                avg_scores.append(df[score_col].mean())
            else:
                avg_scores.append(None)
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=avg_scores,
            mode='lines+markers',
            name='Average',
            line=dict(color='blue', width=4),
            marker=dict(size=8),
            hovertemplate='<b>Average Score</b><br>' +
                        'Epoch: %{x}<br>' +
                        'Score: %{y:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Individual Prompt Trajectories Across Training Epochs ({classifier.upper()})',
            xaxis_title='Training Epoch',
            yaxis_title='Toxicity Score',
            height=600,
            hovermode='closest'
        )
        
        fig.add_annotation(
            text="Green lines = Improved prompts<br>Red lines = Regressed prompts<br>Gray lines = No significant change",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        )
        
        self.log_plot_to_wandb(fig, "individual_prompt_trajectories", 
                             "Individual prompt toxicity trajectories across training epochs")
    
    def create_statistical_analysis(self, df: pd.DataFrame):
        """Create statistical analysis and log to WandB."""
        logger.info("📊 Creating statistical analysis...")
        
        # Get model names
        model_cols = [col for col in df.columns if col.startswith('output_') and not col.endswith('_score')]
        models = [col.replace('output_', '') for col in model_cols]
        
        # Get classifier names
        classifier_cols = [col for col in df.columns if col.endswith('_score') and 'base' in col]
        classifiers = [col.replace('output_base_', '').replace('_score', '') for col in classifier_cols]
        
        # 1. Summary statistics table
        stats_data = []
        
        for model in models:
            model_stats = {'model': model}
            
            # Calculate statistics for each classifier
            for classifier in classifiers:
                delta_col = f'delta_{model}_vs_base_{classifier}'
                if delta_col in df.columns:
                    delta_data = df[delta_col].dropna()
                    
                    model_stats[f'{classifier}_mean'] = delta_data.mean()
                    model_stats[f'{classifier}_std'] = delta_data.std()
                    model_stats[f'{classifier}_positive_count'] = (delta_data > 0.01).sum()
                    model_stats[f'{classifier}_negative_count'] = (delta_data < -0.01).sum()
                    model_stats[f'{classifier}_total_count'] = len(delta_data)
            
            # Overall statistics
            all_improvements = []
            for classifier in classifiers:
                delta_col = f'delta_{model}_vs_base_{classifier}'
                if delta_col in df.columns:
                    all_improvements.extend(df[delta_col].dropna().tolist())
            
            if all_improvements:
                model_stats['overall_mean'] = np.mean(all_improvements)
                model_stats['overall_std'] = np.std(all_improvements)
                model_stats['overall_positive_rate'] = np.mean([x > 0.01 for x in all_improvements])
            
            stats_data.append(model_stats)
        
        stats_df = pd.DataFrame(stats_data)
        
        # Log statistics table
        self.log_dataframe_as_table(stats_df, "statistical_summary", 
                                  "Comprehensive statistical analysis of model improvements")
        
        # 2. Significance testing visualization
        fig = go.Figure()
        
        for i, model in enumerate(models):
            delta_col = f'delta_{model}_vs_base_roberta_score'
            if delta_col in df.columns:
                delta_data = df[delta_col].dropna()
                
                if len(delta_data) > 0:
                    mean_val = delta_data.mean()
                    std_val = delta_data.std()
                    n = len(delta_data)
                    
                    # Calculate 95% confidence interval
                    ci = 1.96 * std_val / np.sqrt(n)
                    
                    fig.add_trace(go.Scatter(
                        x=[model],
                        y=[mean_val],
                        error_y=dict(type='data', array=[ci], visible=True),
                        mode='markers',
                        marker=dict(size=10, color=self.model_colors.get(model, '#95a5a6')),
                        name=model,
                        hovertemplate=f'<b>{model}</b><br>' +
                                    f'Mean: %{{y:.4f}}<br>' +
                                    f'95% CI: ±{ci:.4f}<br>' +
                                    f'Sample Size: {n}<br>' +
                                    '<extra></extra>'
                    ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2,
                     annotation_text="No Change")
        
        fig.update_layout(
            title='Statistical Significance: Mean Improvement with 95% Confidence Intervals',
            xaxis_title='Models',
            yaxis_title='Mean Improvement (RoBERTa)',
            height=500,
            showlegend=False
        )
        
        self.log_plot_to_wandb(fig, "statistical_significance", 
                             "Statistical significance analysis with confidence intervals")
    
    def create_text_type_analysis(self, df: pd.DataFrame):
        """Create interactive text type analysis plots."""
        logger.info("📝 Creating text type analysis plots...")
        
        # Compare prompt vs output vs full text for different models
        fig = go.Figure()
        
        # Get base and final model names
        model_cols = [col for col in df.columns if col.startswith('output_') and not col.endswith('_score')]
        models = [col.replace('output_', '') for col in model_cols]
        
        if len(models) >= 2:
            base_model = 'base'
            final_model = models[-1]  # Assume last model is final
            
            for model in [base_model, final_model]:
                # Get data for each text type
                prompt_data = df['prompt_roberta_score'].dropna() if 'prompt_roberta_score' in df.columns else pd.Series()
                output_data = df[f'output_{model}_roberta_score'].dropna() if f'output_{model}_roberta_score' in df.columns else pd.Series()
                full_data = df[f'full_{model}_roberta_score'].dropna() if f'full_{model}_roberta_score' in df.columns else pd.Series()
                
                # Add violin plots
                if len(prompt_data) > 0:
                    fig.add_trace(go.Violin(
                        y=prompt_data,
                        name=f'Prompt Only',
                        side='negative' if model == base_model else 'positive',
                        line_color='#95a5a6',
                        fillcolor='rgba(149, 165, 166, 0.5)',
                        showlegend=(model == base_model),
                        hovertemplate=f'Prompt Only<br>Score: %{{y:.4f}}<extra></extra>'
                    ))
                
                if len(output_data) > 0:
                    fig.add_trace(go.Violin(
                        y=output_data,
                        name=f'Output Only ({model})',
                        side='negative' if model == base_model else 'positive',
                        line_color=self.model_colors.get(model, '#95a5a6'),
                        fillcolor=f'rgba({",".join(map(str, [52, 73, 94] if model == "base" else [155, 89, 182]))}, 0.5)',
                        hovertemplate=f'Output ({model})<br>Score: %{{y:.4f}}<extra></extra>'
                    ))
                
                if len(full_data) > 0:
                    fig.add_trace(go.Violin(
                        y=full_data,
                        name=f'Full Text ({model})',
                        side='negative' if model == base_model else 'positive',
                        line_color='#34495e',
                        fillcolor='rgba(52, 73, 94, 0.5)',
                        showlegend=(model == base_model),
                        hovertemplate=f'Full Text ({model})<br>Score: %{{y:.4f}}<extra></extra>'
                    ))
            
            fig.update_layout(
                title='Text Type Comparison: Distribution of Toxicity Scores',
                yaxis_title='Toxicity Score',
                violinmode='overlay',
                height=600,
                showlegend=True
            )
            
            self.log_plot_to_wandb(fig, "text_type_comparison", 
                                 "Interactive comparison of toxicity across different text types")
    
    def create_advanced_interactive_dashboard(self, df: pd.DataFrame):
        """Create advanced interactive dashboard with custom controls."""
        logger.info("🎛️ Creating advanced interactive dashboard...")
        
        # Get model names
        model_cols = [col for col in df.columns if col.startswith('output_') and not col.endswith('_score')]
        models = [col.replace('output_', '') for col in model_cols]
        
        # Get classifier names
        classifier_cols = [col for col in df.columns if col.endswith('_score') and 'base' in col]
        classifiers = [col.replace('output_base_', '').replace('_score', '') for col in classifier_cols]
        
        if len(models) >= 3 and len(classifiers) >= 2:
            # 1. Multi-dimensional analysis plot
            fig = go.Figure()
            
            # Create 3D scatter plot with multiple dimensions
            for model in models[:3]:  # Use first 3 models for 3D plot
                roberta_col = f'delta_{model}_vs_base_roberta_score'
                dynabench_col = f'delta_{model}_vs_base_dynabench_score'
                
                if roberta_col in df.columns and dynabench_col in df.columns:
                    mask = ~(df[roberta_col].isna() | df[dynabench_col].isna())
                    
                    # Use a third dimension if available
                    third_dim_col = None
                    for col in df.columns:
                        if col.startswith(f'delta_{model}_vs_base_') and col not in [roberta_col, dynabench_col]:
                            third_dim_col = col
                            break
                    
                    if third_dim_col and third_dim_col in df.columns:
                        mask = mask & ~df[third_dim_col].isna()
                        z_data = df[mask][third_dim_col]
                    else:
                        # Use a derived metric as third dimension
                        z_data = df[mask][roberta_col] + df[mask][dynabench_col]
                    
                    fig.add_trace(go.Scatter3d(
                        x=df[mask][roberta_col],
                        y=df[mask][dynabench_col],
                        z=z_data,
                        mode='markers',
                        name=model,
                        marker=dict(
                            size=4,
                            color=self.model_colors.get(model, '#95a5a6'),
                            opacity=0.6
                        ),
                        hovertemplate=f'<b>{model}</b><br>' +
                                    'RoBERTa: %{x:.4f}<br>' +
                                    'DynaBench: %{y:.4f}<br>' +
                                    'Third Dim: %{z:.4f}<br>' +
                                    '<extra></extra>',
                        visible=True if model == models[-1] else 'legendonly'
                    ))
            
            fig.update_layout(
                title='3D Analysis: Multi-Classifier Improvement Space',
                scene=dict(
                    xaxis_title='RoBERTa Improvement',
                    yaxis_title='DynaBench Improvement',
                    zaxis_title='Third Dimension',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                height=700
            )
            
            self.log_plot_to_wandb(fig, "3d_multi_classifier_analysis", 
                                 "3D analysis of improvements across multiple classifiers")
    
    def _log_comprehensive_metrics(self, df: pd.DataFrame, metrics: Dict[str, Any]):
        """Log basic metrics to WandB from clean datasets."""
        if not self.wandb_run:
            return
        
        models, classifiers, toxic_bert_categories = self._detect_models_and_classifiers(df)
        
        # Ensure we have at least one classifier
        if not classifiers:
            logger.warning("⚠️ No classifiers detected, using default classifiers...")
            classifiers = {'toxic_bert', 'roberta_toxicity', 'dynabench_hate'}
        
        logger.info(f"📊 Logging basic metrics for {len(classifiers)} classifiers")
        
        # Log basic statistics for each classifier
        classifier_metrics = {}
        for classifier in classifiers:
            # Check for different text types
            text_types = ['prompt', 'output', 'full_text']
            
            for text_type in text_types:
                col_name = f'{text_type}_{classifier}_results'
                
                if col_name in df.columns:
                    # Extract scores from dictionary results
                    scores = []
                    
                    for result in df[col_name]:
                        if isinstance(result, dict):
                            # For toxic_bert, use 'toxic' category
                            if classifier == 'toxic_bert':
                                score = result.get('toxic', 0.0)
                            elif classifier == 'roberta_toxicity':
                                score = result.get('toxic', 0.0)
                            elif classifier == 'dynabench_hate':
                                score = result.get('hate', 0.0)
                            else:
                                # Use the first available score
                                score = next(iter(result.values()), 0.0)
                            
                            scores.append(score)
                    
                    if scores:
                        classifier_metrics[f"{text_type}_{classifier}_mean"] = np.mean(scores)
                        classifier_metrics[f"{text_type}_{classifier}_std"] = np.std(scores)
                        classifier_metrics[f"{text_type}_{classifier}_min"] = np.min(scores)
                        classifier_metrics[f"{text_type}_{classifier}_max"] = np.max(scores)
                        classifier_metrics[f"{text_type}_{classifier}_count"] = len(scores)
        
        if classifier_metrics:
            wandb.log(classifier_metrics)
            logger.info(f"✅ Logged {len(classifier_metrics)} basic metrics to WandB")
        
        # Log Toxic-BERT category metrics if available
        if toxic_bert_categories:
            toxic_bert_metrics = {}
            
            for category in toxic_bert_categories:
                # Look for Toxic-BERT results
                col_name = f'output_toxic_bert_results'
                
                if col_name in df.columns:
                    category_scores = []
                    
                    for result in df[col_name]:
                        if isinstance(result, dict):
                            score = result.get(category, 0.0)
                            category_scores.append(score)
                    
                    if category_scores:
                        toxic_bert_metrics[f"toxic_bert_{category}_mean"] = np.mean(category_scores)
                        toxic_bert_metrics[f"toxic_bert_{category}_std"] = np.std(category_scores)
                        toxic_bert_metrics[f"toxic_bert_{category}_min"] = np.min(category_scores)
                        toxic_bert_metrics[f"toxic_bert_{category}_max"] = np.max(category_scores)
            
            if toxic_bert_metrics:
                wandb.log(toxic_bert_metrics)
                logger.info(f"✅ Logged {len(toxic_bert_metrics)} Toxic-BERT category metrics to WandB")
        
        # Log basic dataset information
        basic_metrics = {
            "total_samples": len(df),
            "dataframe_shape": f"{df.shape[0]}x{df.shape[1]}",
            "total_classifiers": len(classifiers),
            "total_toxic_bert_categories": len(toxic_bert_categories)
        }
        
        if 'model' in df.columns:
            unique_models = df['model'].unique()
            basic_metrics["unique_models"] = list(unique_models)
            basic_metrics["total_models"] = len(unique_models)
        
        wandb.log(basic_metrics)
        logger.info("✅ Logged basic dataset information to WandB")
    
    def create_toxic_bert_category_analysis(self, df: pd.DataFrame):
        """Create comprehensive Toxic-BERT category analysis plots."""
        logger.info("🔍 Creating Toxic-BERT category analysis...")
        
        models, classifiers, toxic_bert_categories = self._detect_models_and_classifiers(df)
        
        if not toxic_bert_categories:
            logger.info("No Toxic-BERT categories found, skipping category analysis")
            return
        
        try:
            # Create category-specific visualizations
            self._create_toxic_bert_scatter_plots(df, models, toxic_bert_categories)
            self._create_toxic_bert_heatmap(df, models, toxic_bert_categories)
            self._create_toxic_bert_progression(df, models, toxic_bert_categories)
            self._create_toxic_bert_delta_plots(df, models, toxic_bert_categories)
            
            logger.info("✅ Toxic-BERT category analysis completed")
            
        except Exception as e:
            logger.error(f"❌ Error in Toxic-BERT category analysis: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _create_toxic_bert_scatter_plots(self, df: pd.DataFrame, models: List[str], toxic_bert_categories: List[str]):
        """Create scatter plots for each Toxic-BERT category."""
        logger.info("📈 Creating Toxic-BERT category scatter plots...")
        
        comparison_models = [m for m in models if m != 'base']
        
        for category in toxic_bert_categories:
            fig = go.Figure()
            
            base_col = f'base_toxic_bert_{category}_score'
            
            if base_col not in df.columns:
                logger.warning(f"⚠️ Skipping {category} - base column not found")
                continue
            
            for model in comparison_models:
                model_col = f'{model}_toxic_bert_{category}_score'
                
                if model_col in df.columns:
                    # Remove NaN values
                    mask = ~(df[base_col].isna() | df[model_col].isna())
                    x_data = df[mask][base_col]
                    y_data = df[mask][model_col]
                    
                    if len(x_data) > 0:
                        improvement = (x_data - y_data).mean()
                        model_color = self.model_colors.get(model, '#7f8c8d')
                        
                        fig.add_trace(go.Scatter(
                            x=x_data,
                            y=y_data,
                            mode='markers',
                            name=f'{model.replace("detox_", "")} (Δ={improvement:.4f})',
                            marker=dict(color=model_color, size=4, opacity=0.6),
                            hovertemplate=f'<b>{model} - {category}</b><br>Base: %{{x:.4f}}<br>{model}: %{{y:.4f}}<extra></extra>',
                            visible=True if 'epoch_100' in model else 'legendonly'
                        ))
            
            # Add diagonal line
            if len(df[base_col].dropna()) > 0:
                score_cols = [col for col in df.columns if col.endswith(f'{category}_score')]
                all_scores = pd.concat([df[col].dropna() for col in score_cols if col in df.columns])
                if len(all_scores) > 0:
                    max_val = all_scores.max()
                    min_val = all_scores.min()
                    
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val], y=[min_val, max_val],
                        mode='lines', name='No Change',
                        line=dict(color='red', dash='dash', width=2),
                        hoverinfo='none', showlegend=True
                    ))
                    
                    fig.update_layout(
                        title=f'Toxic-BERT {category.title()}: Base vs Fine-tuned Models',
                        xaxis_title='Base Model Score',
                        yaxis_title='Fine-tuned Model Score',
                        height=600, showlegend=True,
                        xaxis=dict(range=[min_val, max_val]),
                        yaxis=dict(range=[min_val, max_val])
                    )
                    
                    self.log_plot_to_wandb(fig, f"scatter_toxic_bert_{category}", f"Scatter plot for Toxic-BERT {category}")
    
    def _create_toxic_bert_heatmap(self, df: pd.DataFrame, models: List[str], toxic_bert_categories: List[str]):
        """Create heatmap comparing Toxic-BERT categories across models."""
        logger.info("🏆 Creating Toxic-BERT category heatmap...")
        
        comparison_models = [m for m in models if m != 'base']
        comparison_data = []
        
        for model in comparison_models:
            row_data = {'model': model.replace('detox_', '')}
            
            for category in toxic_bert_categories:
                delta_col = f'delta_{model}_vs_base_toxic_bert_{category}'
                if delta_col in df.columns:
                    avg_improvement = df[delta_col].mean()
                    row_data[category] = avg_improvement
                else:
                    row_data[category] = 0.0
            
            comparison_data.append(row_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create heatmap
        z_data = comparison_df[toxic_bert_categories].values
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=[c.title() for c in toxic_bert_categories],
            y=comparison_df['model'],
            colorscale='RdYlGn',
            colorbar=dict(title="Improvement"),
            hovertemplate='Model: %{y}<br>Category: %{x}<br>Improvement: %{z:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Toxic-BERT Categories: Model Performance Heatmap',
            height=400
        )
        
        self.log_plot_to_wandb(fig, "heatmap_toxic_bert_categories", "Toxic-BERT category comparison heatmap")
    
    def _create_toxic_bert_progression(self, df: pd.DataFrame, models: List[str], toxic_bert_categories: List[str]):
        """Create progression plots for Toxic-BERT categories."""
        logger.info("📈 Creating Toxic-BERT category progression plots...")
        
        # Extract epoch information
        epoch_models = []
        for model in models:
            if model == 'base':
                epoch_models.append((0, model))
            elif 'epoch_' in model:
                try:
                    epoch = int(model.split('epoch_')[1])
                    epoch_models.append((epoch, model))
                except:
                    continue
        
        epoch_models.sort(key=lambda x: x[0])
        epochs = [epoch for epoch, model in epoch_models]
        
        # Create subplot for all categories
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[c.title() for c in toxic_bert_categories],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        for i, category in enumerate(toxic_bert_categories):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            improvements = []
            
            for epoch, model in epoch_models:
                if model == 'base':
                    improvements.append(0.0)
                else:
                    delta_col = f'delta_{model}_vs_base_toxic_bert_{category}'
                    if delta_col in df.columns:
                        avg_improvement = df[delta_col].mean()
                        improvements.append(avg_improvement)
                    else:
                        improvements.append(0)
            
            category_color = self.category_colors.get(category, '#3498db')
            
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=improvements,
                    mode='lines+markers',
                    name=category.title(),
                    line=dict(color=category_color, width=2),
                    marker=dict(size=6),
                    hovertemplate=f'<b>{category}</b><br>Epoch: %{{x}}<br>Avg Improvement: %{{y:.4f}}<extra></extra>',
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=1, row=row, col=col)
        
        fig.update_layout(
            title='Toxic-BERT Categories: Training Progression',
            height=800,
            showlegend=False
        )
        
        # Update axes labels
        for i in range(1, 3):
            for j in range(1, 4):
                fig.update_xaxes(title_text="Epoch", row=i, col=j)
                fig.update_yaxes(title_text="Improvement", row=i, col=j)
        
        self.log_plot_to_wandb(fig, "progression_toxic_bert_categories", "Toxic-BERT category progression")
    
    def _create_toxic_bert_delta_plots(self, df: pd.DataFrame, models: List[str], toxic_bert_categories: List[str]):
        """Create improvement distribution plots for Toxic-BERT categories."""
        logger.info("📊 Creating Toxic-BERT category delta plots...")
        
        comparison_models = [m for m in models if m != 'base']
        
        for category in toxic_bert_categories:
            fig = go.Figure()
            
            for model in comparison_models:
                delta_col = f'delta_{model}_vs_base_toxic_bert_{category}'
                
                if delta_col in df.columns:
                    delta_data = df[delta_col].dropna()
                    
                    if len(delta_data) > 0:
                        mean_improvement = delta_data.mean()
                        model_color = self.model_colors.get(model, '#7f8c8d')
                        
                        fig.add_trace(go.Histogram(
                            x=delta_data,
                            name=f'{model.replace("detox_", "")} (μ={mean_improvement:.4f})',
                            opacity=0.7, marker_color=model_color, nbinsx=30,
                            hovertemplate=f'<b>{model} - {category}</b><br>Improvement: %{{x:.4f}}<br>Count: %{{y}}<extra></extra>'
                        ))
            
            fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=2)
            
            fig.update_layout(
                title=f'Toxic-BERT {category.title()}: Improvement Distribution',
                xaxis_title='Toxicity Reduction (positive = better)',
                yaxis_title='Frequency', barmode='overlay',
                height=600, showlegend=True
            )
            
            self.log_plot_to_wandb(fig, f"delta_toxic_bert_{category}", f"Improvement distribution for Toxic-BERT {category}")

    def cleanup(self):
        """Clean up resources."""
        if self.wandb_run:
            wandb.finish() 