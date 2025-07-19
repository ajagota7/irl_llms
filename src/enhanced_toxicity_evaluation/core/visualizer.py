"""
Enhanced visualization tools for toxicity evaluation results.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


class ToxicityVisualizer:
    """Enhanced visualization tools for toxicity evaluation results."""
    
    def __init__(self, results_df: pd.DataFrame, output_dir: Optional[Path] = None):
        """Initialize visualizer with results DataFrame."""
        self.df = results_df
        self.output_dir = output_dir or Path("visualizations")
        self.output_dir.mkdir(exist_ok=True)
        
        # Extract available models and classifiers
        self.models = self._extract_models()
        self.classifiers = self._extract_classifiers()
        
        logger.info(f"Visualizer initialized with {len(self.df)} samples")
    
    def _extract_models(self) -> List[str]:
        """Extract model names from DataFrame columns."""
        models = set()
        for col in self.df.columns:
            if col.startswith("output_") and not col.endswith("_score"):
                parts = col.split("_")
                if len(parts) >= 2:
                    models.add(parts[1])
        return sorted(list(models))
    
    def _extract_classifiers(self) -> List[str]:
        """Extract classifier names from DataFrame columns."""
        classifiers = set()
        for col in self.df.columns:
            if col.endswith("_score") and not col.startswith("delta_"):
                parts = col.replace("_score", "").split("_")
                if len(parts) >= 2:
                    classifiers.add(parts[-1])
        return sorted(list(classifiers))
    
    def plot_toxicity_distributions(self, save_html: bool = True) -> go.Figure:
        """Plot toxicity score distributions for each model and classifier."""
        n_classifiers = len(self.classifiers)
        n_models = len(self.models)
        
        fig = make_subplots(
            rows=n_classifiers, 
            cols=n_models,
            subplot_titles=[f"{clf} - {model}" for clf in self.classifiers for model in self.models],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, classifier in enumerate(self.classifiers):
            for j, model in enumerate(self.models):
                col_name = f"output_{model}_{classifier}_score"
                if col_name in self.df.columns:
                    fig.add_trace(
                        go.Histogram(
                            x=self.df[col_name], 
                            name=f"{classifier}_{model}",
                            nbinsx=20,
                            marker_color=colors[j % len(colors)],
                            showlegend=False
                        ),
                        row=i+1, col=j+1
                    )
        
        fig.update_layout(
            height=300*n_classifiers,
            title="Toxicity Score Distributions by Model and Classifier",
            showlegend=False
        )
        
        if save_html:
            output_path = self.output_dir / "toxicity_distributions.html"
            fig.write_html(output_path)
            logger.info(f"Toxicity distributions saved to {output_path}")
        
        return fig
    
    def plot_model_comparison_scatter(self, model: str, classifier: str = "roberta", 
                                    save_html: bool = True) -> go.Figure:
        """Create scatter plot comparing model vs base."""
        base_col = f"output_base_{classifier}_score"
        model_col = f"output_{model}_{classifier}_score"
        
        if not all(col in self.df.columns for col in [base_col, model_col]):
            raise ValueError(f"Required columns not found: {base_col}, {model_col}")
        
        fig = px.scatter(
            self.df, 
            x=base_col, 
            y=model_col,
            title=f"{classifier.title()} Scores: Base vs {model}",
            labels={base_col: f"Base Model Toxicity", model_col: f"{model} Model Toxicity"},
            opacity=0.6
        )
        
        # Add diagonal line
        max_val = max(self.df[base_col].max(), self.df[model_col].max())
        fig.add_shape(
            type="line", 
            line=dict(dash="dash", color="red"),
            x0=0, x1=max_val, y0=0, y1=max_val
        )
        
        # Add improvement/regression regions
        fig.add_annotation(
            x=max_val*0.8, y=max_val*0.2,
            text="Improvement<br>(Below diagonal)",
            showarrow=False,
            bgcolor="lightgreen",
            opacity=0.7
        )
        
        if save_html:
            output_path = self.output_dir / f"scatter_{classifier}_{model}_vs_base.html"
            fig.write_html(output_path)
            logger.info(f"Scatter plot saved to {output_path}")
        
        return fig
    
    def plot_delta_analysis(self, model: str, save_html: bool = True) -> go.Figure:
        """Plot improvement/regression analysis for a model."""
        delta_cols = [col for col in self.df.columns if col.startswith(f"delta_{model}_vs_base_")]
        
        if not delta_cols:
            raise ValueError(f"No delta columns found for model {model}")
        
        fig = make_subplots(
            rows=len(delta_cols), cols=1,
            subplot_titles=[col.replace(f"delta_{model}_vs_base_", "").replace("_score", "") for col in delta_cols],
            vertical_spacing=0.1
        )
        
        for i, col in enumerate(delta_cols):
            # Create histogram of improvements
            fig.add_trace(
                go.Histogram(
                    x=self.df[col],
                    name=col.replace(f"delta_{model}_vs_base_", ""),
                    nbinsx=30,
                    marker_color='skyblue',
                    showlegend=False
                ),
                row=i+1, col=1
            )
            
            # Add vertical line at zero
            fig.add_vline(
                x=0, line_dash="dash", line_color="red",
                row=i+1, col=1
            )
            
            # Add improvement/regression annotations
            improvements = (self.df[col] > 0.05).sum()
            regressions = (self.df[col] < -0.05).sum()
            
            fig.add_annotation(
                x=0.1, y=0.9,
                text=f"Improvements: {improvements}<br>Regressions: {regressions}",
                showarrow=False,
                xref=f"x{i+1}", yref=f"y{i+1}",
                bgcolor="white",
                bordercolor="black"
            )
        
        fig.update_layout(
            height=300*len(delta_cols),
            title=f"Toxicity Improvement Analysis: {model} vs Base",
            xaxis_title="Improvement (positive = better)"
        )
        
        if save_html:
            output_path = self.output_dir / f"delta_analysis_{model}.html"
            fig.write_html(output_path)
            logger.info(f"Delta analysis saved to {output_path}")
        
        return fig
    
    def create_comprehensive_dashboard(self, save_html: bool = True) -> go.Figure:
        """Create comprehensive dashboard with multiple visualizations."""
        # Calculate summary statistics for dashboard
        model_stats = []
        for model in self.models:
            if model != "base":
                for classifier in self.classifiers:
                    base_col = f"output_base_{classifier}_score"
                    model_col = f"output_{model}_{classifier}_score"
                    delta_col = f"delta_{model}_vs_base_{classifier}_score"
                    
                    if all(col in self.df.columns for col in [base_col, model_col, delta_col]):
                        model_stats.append({
                            "model": model,
                            "classifier": classifier,
                            "base_mean": self.df[base_col].mean(),
                            "model_mean": self.df[model_col].mean(),
                            "improvement": self.df[delta_col].mean(),
                            "improvements_count": (self.df[delta_col] > 0.05).sum(),
                            "regressions_count": (self.df[delta_col] < -0.05).sum()
                        })
        
        stats_df = pd.DataFrame(model_stats)
        
        if len(stats_df) == 0:
            logger.warning("No data available for dashboard")
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Average Toxicity by Model", "Improvement Distribution", 
                          "Improvements vs Regressions", "Model Performance Heatmap"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Average toxicity by model
        for classifier in stats_df['classifier'].unique():
            classifier_data = stats_df[stats_df['classifier'] == classifier]
            fig.add_trace(
                go.Bar(
                    x=classifier_data['model'],
                    y=classifier_data['model_mean'],
                    name=f"{classifier} (Model)",
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Plot 2: Improvement distribution
        fig.add_trace(
            go.Histogram(
                x=stats_df['improvement'],
                name="Improvement Distribution",
                nbinsx=20,
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Plot 3: Improvements vs Regressions
        fig.add_trace(
            go.Scatter(
                x=stats_df['improvements_count'],
                y=stats_df['regressions_count'],
                mode='markers+text',
                text=stats_df['model'],
                textposition="top center",
                name="Models",
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Plot 4: Performance heatmap (using bar chart as approximation)
        pivot_data = stats_df.pivot(index='model', columns='classifier', values='improvement')
        if not pivot_data.empty:
            fig.add_trace(
                go.Heatmap(
                    z=pivot_data.values,
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    colorscale='RdYlGn',
                    showscale=True,
                    name="Improvement Heatmap"
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title="Comprehensive Toxicity Evaluation Dashboard",
            showlegend=True
        )
        
        if save_html:
            output_path = self.output_dir / "comprehensive_dashboard.html"
            fig.write_html(output_path)
            logger.info(f"Comprehensive dashboard saved to {output_path}")
        
        return fig
    
    def plot_toxic_bert_categories(self, model: str, save_html: bool = True) -> go.Figure:
        """Plot toxic-bert category analysis."""
        # Find toxic-bert columns
        toxic_bert_cols = [col for col in self.df.columns if "toxic_bert_" in col and f"output_{model}" in col]
        
        if not toxic_bert_cols:
            logger.warning(f"No toxic-bert columns found for model {model}")
            return go.Figure()
        
        # Extract categories
        categories = []
        base_scores = []
        model_scores = []
        
        for col in toxic_bert_cols:
            category = col.split("toxic_bert_")[-1]
            base_col = f"output_base_toxic_bert_{category}"
            
            if base_col in self.df.columns:
                categories.append(category)
                base_scores.append(self.df[base_col].mean())
                model_scores.append(self.df[col].mean())
        
        if not categories:
            logger.warning(f"No matching base columns found for toxic-bert analysis")
            return go.Figure()
        
        fig = go.Figure()
        
        # Add base model scores
        fig.add_trace(go.Bar(
            x=categories,
            y=base_scores,
            name="Base Model",
            marker_color='lightcoral'
        ))
        
        # Add fine-tuned model scores
        fig.add_trace(go.Bar(
            x=categories,
            y=model_scores,
            name=f"{model} Model",
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title=f"Toxic-BERT Category Analysis: {model} vs Base",
            xaxis_title="Toxicity Categories",
            yaxis_title="Average Score",
            barmode='group'
        )
        
        if save_html:
            output_path = self.output_dir / f"toxic_bert_categories_{model}.html"
            fig.write_html(output_path)
            logger.info(f"Toxic-BERT category analysis saved to {output_path}")
        
        return fig 