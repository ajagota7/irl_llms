"""
Comprehensive visualization manager for the Enhanced Toxicity Evaluation Pipeline.
Provides interactive plots, statistical analysis, and WandB integration.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wandb
from typing import Dict, List, Optional, Any, Tuple
import warnings
from pathlib import Path
import logging

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


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
            self.wandb_run = wandb.init(
                project=self.wandb_config.get("wandb_project", "toxicity-evaluation"),
                entity=self.wandb_config.get("wandb_entity"),
                name=self.experiment_config.get("name"),
                config=self.config,
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
    
    def create_comprehensive_visualizations(self, results_df: pd.DataFrame, metrics: Dict[str, Any]):
        """Create all comprehensive visualizations and log to WandB."""
        logger.info("🎨 Creating comprehensive visualizations...")
        
        try:
            # Log the full CSV to WandB
            self._log_full_csv_to_wandb(results_df)
            
            # Create all visualization types
            self.create_interactive_scatter_plots(results_df)
            self.create_interactive_delta_plots(results_df)
            self.create_interactive_progression_plots(results_df)
            self.create_model_comparison_plots(results_df)
            self.create_individual_prompt_tracking(results_df)
            self.create_statistical_analysis(results_df)
            self.create_text_type_analysis(results_df)
            self.create_advanced_interactive_dashboard(results_df)
            
            # Log comprehensive metrics
            self._log_comprehensive_metrics(results_df, metrics)
            
            logger.info("✅ All visualizations created and logged successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error during visualization creation: {e}")
            raise
    
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
        
        # Get model names from columns
        model_cols = [col for col in df.columns if col.startswith('output_') and not col.endswith('_score')]
        models = [col.replace('output_', '') for col in model_cols]
        
        # Get classifier names
        classifier_cols = [col for col in df.columns if col.endswith('_score') and 'base' in col]
        classifiers = [col.replace('output_base_', '').replace('_score', '') for col in classifier_cols]
        
        # 1. Main classifiers scatter plots with model selection
        for classifier in classifiers:
            fig = go.Figure()
            
            # Add traces for each model
            for model in models:
                base_col = f'output_base_{classifier}_score'
                model_col = f'output_{model}_{classifier}_score'
                
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
        
        # Get delta columns
        delta_cols = [col for col in df.columns if col.startswith('delta_')]
        
        # Group by classifier
        classifiers = set()
        for col in delta_cols:
            parts = col.split('_')
            if len(parts) >= 4:
                classifier = parts[-2]  # Assuming format: delta_model_vs_base_classifier_score
                classifiers.add(classifier)
        
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
        
        # Extract epoch numbers from model names
        epoch_models = {}
        for col in df.columns:
            if col.startswith('output_') and not col.endswith('_score'):
                model_name = col.replace('output_', '')
                if 'epoch_' in model_name:
                    try:
                        epoch_num = int(model_name.split('_')[1])
                        epoch_models[epoch_num] = model_name
                    except (ValueError, IndexError):
                        continue
        
        if not epoch_models:
            logger.warning("No epoch-based models found for progression plots")
            return
        
        epochs = sorted(epoch_models.keys())
        
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
        
        # Extract epoch numbers from model names
        epoch_models = {}
        for col in df.columns:
            if col.startswith('output_') and not col.endswith('_score'):
                model_name = col.replace('output_', '')
                if 'epoch_' in model_name:
                    try:
                        epoch_num = int(model_name.split('_')[1])
                        epoch_models[epoch_num] = model_name
                    except (ValueError, IndexError):
                        continue
        
        if not epoch_models:
            logger.warning("No epoch-based models found for prompt tracking")
            return
        
        epochs = sorted(epoch_models.keys())
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
                delta_col = f'delta_{model}_vs_base_{classifier}_score'
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
                delta_col = f'delta_{model}_vs_base_{classifier}_score'
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
        """Log comprehensive metrics to WandB."""
        logger.info("📈 Logging comprehensive metrics...")
        
        if not self.wandb_run:
            return
        
        # Get model names
        model_cols = [col for col in df.columns if col.startswith('output_') and not col.endswith('_score')]
        models = [col.replace('output_', '') for col in model_cols]
        
        # Get classifier names
        classifier_cols = [col for col in df.columns if col.endswith('_score') and 'base' in col]
        classifiers = [col.replace('output_base_', '').replace('_score', '') for col in classifier_cols]
        
        # 1. Model performance metrics
        model_metrics = {}
        
        for model in models:
            # Calculate overall performance across all classifiers
            all_improvements = []
            
            for classifier in classifiers:
                delta_col = f'delta_{model}_vs_base_{classifier}_score'
                if delta_col in df.columns:
                    all_improvements.extend(df[delta_col].dropna().tolist())
            
            if all_improvements:
                model_metrics[f"{model}_mean_improvement"] = np.mean(all_improvements)
                model_metrics[f"{model}_std_improvement"] = np.std(all_improvements)
                model_metrics[f"{model}_positive_rate"] = np.mean([x > 0.01 for x in all_improvements])
                model_metrics[f"{model}_strong_positive_rate"] = np.mean([x > 0.1 for x in all_improvements])
                model_metrics[f"{model}_regression_rate"] = np.mean([x < -0.01 for x in all_improvements])
        
        wandb.log(model_metrics)
        
        # 2. Classifier-specific metrics
        classifier_metrics = {}
        
        for classifier in classifiers:
            classifier_improvements = []
            
            for model in models:
                delta_col = f'delta_{model}_vs_base_{classifier}_score'
                if delta_col in df.columns:
                    classifier_improvements.extend(df[delta_col].dropna().tolist())
            
            if classifier_improvements:
                classifier_metrics[f"{classifier}_overall_mean"] = np.mean(classifier_improvements)
                classifier_metrics[f"{classifier}_overall_std"] = np.std(classifier_improvements)
                classifier_metrics[f"{classifier}_best_model"] = max(models, key=lambda m: 
                    df[f'delta_{m}_vs_base_{classifier}_score'].mean() if f'delta_{m}_vs_base_{classifier}_score' in df.columns else -1)
        
        wandb.log(classifier_metrics)
        
        # 3. High-level summary metrics
        summary_metrics = {
            "best_overall_model": max(models, key=lambda m: model_metrics.get(f"{m}_mean_improvement", -1)),
            "most_consistent_model": min(models, key=lambda m: model_metrics.get(f"{m}_std_improvement", float('inf'))),
            "best_classifier": max(classifiers, key=lambda c: classifier_metrics.get(f"{c}_overall_mean", -1))
        }
        
        wandb.log(summary_metrics)
    
    def cleanup(self):
        """Clean up resources."""
        if self.wandb_run:
            wandb.finish() 