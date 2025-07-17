"""
Visualization module for the Enhanced Toxicity Evaluation Pipeline.
Provides comprehensive plotting capabilities including scatter plots, distributions, and comparisons.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

logger = logging.getLogger(__name__)


class ToxicityVisualizer:
    """Comprehensive visualization class for toxicity evaluation results."""
    
    def __init__(self, output_dir: Path, config: Optional[Dict] = None):
        """Initialize the visualizer with output directory and configuration."""
        self.output_dir = output_dir
        self.config = config or {}
        self.plots_dir = output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        self._setup_plotting_style()
        
        logger.info(f"ToxicityVisualizer initialized with plots_dir: {self.plots_dir}")
    
    def _setup_plotting_style(self):
        """Setup matplotlib and seaborn plotting styles."""
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Configure matplotlib
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.dpi'] = 300
    
    def create_scatter_plots(self, results_df: pd.DataFrame, 
                           baseline_model: str = "base") -> Dict[str, str]:
        """Create comprehensive scatter plots comparing models."""
        logger.info("Creating scatter plots...")
        
        plot_paths = {}
        
        # Get model names (excluding baseline)
        model_cols = [col for col in results_df.columns 
                     if col.startswith("output_") and not col.endswith("_score")]
        model_names = [col.replace("output_", "") for col in model_cols]
        model_names = [name for name in model_names if name != baseline_model]
        
        # Get classifier names
        classifier_cols = [col for col in results_df.columns 
                          if col.endswith("_score") and "prompt_" not in col]
        classifier_names = list(set([
            col.split("_")[-2] if col.split("_")[-1] == "score" else col.split("_")[-3]
            for col in classifier_cols
        ]))
        
        # Create scatter plots for each classifier
        for classifier in classifier_names:
            plot_path = self._create_model_comparison_scatter(
                results_df, model_names, baseline_model, classifier
            )
            if plot_path:
                plot_paths[f"scatter_{classifier}"] = plot_path
        
        # Create progression scatter plot (if we have epoch-based models)
        progression_plot = self._create_progression_scatter(results_df, classifier_names)
        if progression_plot:
            plot_paths["scatter_progression"] = progression_plot
        
        # Create correlation scatter matrix
        correlation_plot = self._create_correlation_matrix(results_df, classifier_names)
        if correlation_plot:
            plot_paths["correlation_matrix"] = correlation_plot
        
        logger.info(f"Created {len(plot_paths)} scatter plots")
        return plot_paths
    
    def _create_model_comparison_scatter(self, results_df: pd.DataFrame, 
                                       model_names: List[str], 
                                       baseline_model: str, 
                                       classifier: str) -> Optional[str]:
        """Create scatter plot comparing models against baseline for a specific classifier."""
        try:
            baseline_col = f"output_{baseline_model}_{classifier}_score"
            
            if baseline_col not in results_df.columns:
                logger.warning(f"Baseline column {baseline_col} not found")
                return None
            
            # Create subplot grid
            n_models = len(model_names)
            n_cols = min(3, n_models)
            n_rows = (n_models + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
            if n_models == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            else:
                axes = axes.flatten()
            
            # Create scatter plots for each model
            for i, model_name in enumerate(model_names):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                model_col = f"output_{model_name}_{classifier}_score"
                
                if model_col not in results_df.columns:
                    continue
                
                # Get data
                baseline_scores = results_df[baseline_col].dropna()
                model_scores = results_df[model_col].dropna()
                
                # Align data
                common_indices = baseline_scores.index.intersection(model_scores.index)
                if len(common_indices) == 0:
                    continue
                
                x = baseline_scores.loc[common_indices]
                y = model_scores.loc[common_indices]
                
                # Create scatter plot
                ax.scatter(x, y, alpha=0.6, s=20, edgecolors='none')
                
                # Add diagonal line (y=x)
                min_val = min(x.min(), y.min())
                max_val = max(x.max(), y.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y=x')
                
                # Add regression line
                if len(x) > 1:
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    ax.plot(x, p(x), "g-", alpha=0.8, label=f'R²={np.corrcoef(x, y)[0,1]**2:.3f}')
                
                # Calculate improvement metrics
                improvement = (x - y).mean()
                improved_rate = (y < x).mean()
                
                # Set labels and title
                ax.set_xlabel(f'{baseline_model.title()} Toxicity Score')
                ax.set_ylabel(f'{model_name.title()} Toxicity Score')
                ax.set_title(f'{model_name.title()} vs {baseline_model.title()}\n'
                           f'Improvement: {improvement:.3f}, Better: {improved_rate:.1%}')
                
                # Add legend
                ax.legend()
                
                # Set equal aspect ratio
                ax.set_aspect('equal', adjustable='box')
                
                # Add grid
                ax.grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(len(model_names), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.plots_dir / f"scatter_comparison_{classifier}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error creating model comparison scatter plot: {e}")
            return None
    
    def _create_progression_scatter(self, results_df: pd.DataFrame, 
                                  classifier_names: List[str]) -> Optional[str]:
        """Create scatter plot showing progression across epochs."""
        try:
            # Find epoch-based models
            epoch_models = []
            for col in results_df.columns:
                if col.startswith("output_") and "epoch_" in col and col.endswith("_score"):
                    model_name = col.replace("output_", "").replace("_score", "")
                    classifier = model_name.split("_")[-1]
                    epoch = model_name.split("_")[-2]
                    if epoch.startswith("epoch"):
                        epoch_models.append((model_name, classifier, epoch))
            
            if not epoch_models:
                return None
            
            # Group by classifier
            for classifier in classifier_names:
                classifier_models = [m for m in epoch_models if m[1] == classifier]
                if len(classifier_models) < 2:
                    continue
                
                # Sort by epoch number
                classifier_models.sort(key=lambda x: int(x[2].replace("epoch_", "")))
                
                # Create subplot grid
                n_models = len(classifier_models)
                n_cols = min(3, n_models)
                n_rows = (n_models + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
                if n_models == 1:
                    axes = [axes]
                elif n_rows == 1:
                    axes = axes.reshape(1, -1)
                else:
                    axes = axes.flatten()
                
                # Create progression plots
                for i, (model_name, _, epoch) in enumerate(classifier_models):
                    if i >= len(axes):
                        break
                        
                    ax = axes[i]
                    model_col = f"output_{model_name}_{classifier}_score"
                    
                    if model_col not in results_df.columns:
                        continue
                    
                    scores = results_df[model_col].dropna()
                    
                    # Create histogram
                    ax.hist(scores, bins=30, alpha=0.7, edgecolor='black')
                    ax.axvline(scores.mean(), color='red', linestyle='--', 
                              label=f'Mean: {scores.mean():.3f}')
                    ax.axvline(scores.median(), color='green', linestyle='--', 
                              label=f'Median: {scores.median():.3f}')
                    
                    ax.set_xlabel('Toxicity Score')
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'{epoch.title()} - {classifier.title()}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                # Hide empty subplots
                for i in range(len(classifier_models), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                
                # Save plot
                plot_path = self.plots_dir / f"progression_{classifier}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                return str(plot_path)
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating progression scatter plot: {e}")
            return None
    
    def _create_correlation_matrix(self, results_df: pd.DataFrame, 
                                 classifier_names: List[str]) -> Optional[str]:
        """Create correlation matrix heatmap for all toxicity scores."""
        try:
            # Get all toxicity score columns
            score_cols = [col for col in results_df.columns 
                         if col.endswith("_score") and "prompt_" not in col]
            
            if len(score_cols) < 2:
                return None
            
            # Create correlation matrix
            corr_matrix = results_df[score_cols].corr()
            
            # Create heatmap
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5, fmt='.3f')
            
            plt.title('Toxicity Score Correlation Matrix', fontsize=16, pad=20)
            plt.tight_layout()
            
            # Save plot
            plot_path = self.plots_dir / "correlation_matrix.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error creating correlation matrix: {e}")
            return None
    
    def create_distribution_plots(self, results_df: pd.DataFrame) -> Dict[str, str]:
        """Create distribution plots for toxicity scores."""
        logger.info("Creating distribution plots...")
        
        plot_paths = {}
        
        # Get toxicity score columns
        score_cols = [col for col in results_df.columns 
                     if col.endswith("_score") and "prompt_" not in col]
        
        if not score_cols:
            logger.warning("No toxicity score columns found")
            return plot_paths
        
        # Create overall distribution plot
        overall_plot = self._create_overall_distribution(results_df, score_cols)
        if overall_plot:
            plot_paths["overall_distribution"] = overall_plot
        
        # Create model-wise distribution plots
        model_plots = self._create_model_distributions(results_df, score_cols)
        plot_paths.update(model_plots)
        
        # Create classifier-wise distribution plots
        classifier_plots = self._create_classifier_distributions(results_df, score_cols)
        plot_paths.update(classifier_plots)
        
        logger.info(f"Created {len(plot_paths)} distribution plots")
        return plot_paths
    
    def _create_overall_distribution(self, results_df: pd.DataFrame, 
                                   score_cols: List[str]) -> Optional[str]:
        """Create overall distribution plot for all toxicity scores."""
        try:
            # Prepare data for plotting
            plot_data = []
            for col in score_cols:
                scores = results_df[col].dropna()
                if len(scores) > 0:
                    plot_data.extend([(col, score) for score in scores])
            
            if not plot_data:
                return None
            
            df_plot = pd.DataFrame(plot_data, columns=['Model', 'Score'])
            
            # Create violin plot
            plt.figure(figsize=(15, 8))
            sns.violinplot(data=df_plot, x='Model', y='Score')
            plt.xticks(rotation=45, ha='right')
            plt.title('Overall Toxicity Score Distribution', fontsize=16)
            plt.tight_layout()
            
            # Save plot
            plot_path = self.plots_dir / "overall_distribution.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error creating overall distribution plot: {e}")
            return None
    
    def _create_model_distributions(self, results_df: pd.DataFrame, 
                                  score_cols: List[str]) -> Dict[str, str]:
        """Create distribution plots grouped by model."""
        plot_paths = {}
        
        try:
            # Group columns by model
            model_groups = {}
            for col in score_cols:
                parts = col.split("_")
                if len(parts) >= 3:
                    model_name = parts[1]  # output_{model}_{classifier}_score
                    if model_name not in model_groups:
                        model_groups[model_name] = []
                    model_groups[model_name].append(col)
            
            # Create plots for each model
            for model_name, cols in model_groups.items():
                if len(cols) < 2:
                    continue
                
                # Create subplot grid
                n_cols = min(3, len(cols))
                n_rows = (len(cols) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
                if len(cols) == 1:
                    axes = [axes]
                elif n_rows == 1:
                    axes = axes.reshape(1, -1)
                else:
                    axes = axes.flatten()
                
                # Create plots for each classifier
                for i, col in enumerate(cols):
                    if i >= len(axes):
                        break
                        
                    ax = axes[i]
                    scores = results_df[col].dropna()
                    
                    if len(scores) > 0:
                        # Create histogram with KDE
                        ax.hist(scores, bins=30, alpha=0.7, density=True, edgecolor='black')
                        sns.kdeplot(scores, ax=ax, color='red', linewidth=2)
                        
                        # Add statistics
                        mean_score = scores.mean()
                        median_score = scores.median()
                        ax.axvline(mean_score, color='blue', linestyle='--', 
                                  label=f'Mean: {mean_score:.3f}')
                        ax.axvline(median_score, color='green', linestyle='--', 
                                  label=f'Median: {median_score:.3f}')
                        
                        classifier_name = col.split("_")[-2]
                        ax.set_xlabel('Toxicity Score')
                        ax.set_ylabel('Density')
                        ax.set_title(f'{model_name.title()} - {classifier_name.title()}')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                
                # Hide empty subplots
                for i in range(len(cols), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                
                # Save plot
                plot_path = self.plots_dir / f"distribution_{model_name}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                plot_paths[f"distribution_{model_name}"] = str(plot_path)
            
            return plot_paths
            
        except Exception as e:
            logger.error(f"Error creating model distribution plots: {e}")
            return {}
    
    def _create_classifier_distributions(self, results_df: pd.DataFrame, 
                                       score_cols: List[str]) -> Dict[str, str]:
        """Create distribution plots grouped by classifier."""
        plot_paths = {}
        
        try:
            # Group columns by classifier
            classifier_groups = {}
            for col in score_cols:
                parts = col.split("_")
                if len(parts) >= 3:
                    classifier_name = parts[-2]  # output_{model}_{classifier}_score
                    if classifier_name not in classifier_groups:
                        classifier_groups[classifier_name] = []
                    classifier_groups[classifier_name].append(col)
            
            # Create plots for each classifier
            for classifier_name, cols in classifier_groups.items():
                if len(cols) < 2:
                    continue
                
                # Create subplot grid
                n_cols = min(3, len(cols))
                n_rows = (len(cols) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
                if len(cols) == 1:
                    axes = [axes]
                elif n_rows == 1:
                    axes = axes.reshape(1, -1)
                else:
                    axes = axes.flatten()
                
                # Create plots for each model
                for i, col in enumerate(cols):
                    if i >= len(axes):
                        break
                        
                    ax = axes[i]
                    scores = results_df[col].dropna()
                    
                    if len(scores) > 0:
                        # Create histogram with KDE
                        ax.hist(scores, bins=30, alpha=0.7, density=True, edgecolor='black')
                        sns.kdeplot(scores, ax=ax, color='red', linewidth=2)
                        
                        # Add statistics
                        mean_score = scores.mean()
                        median_score = scores.median()
                        ax.axvline(mean_score, color='blue', linestyle='--', 
                                  label=f'Mean: {mean_score:.3f}')
                        ax.axvline(median_score, color='green', linestyle='--', 
                                  label=f'Median: {median_score:.3f}')
                        
                        model_name = col.split("_")[1]
                        ax.set_xlabel('Toxicity Score')
                        ax.set_ylabel('Density')
                        ax.set_title(f'{model_name.title()} - {classifier_name.title()}')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                
                # Hide empty subplots
                for i in range(len(cols), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                
                # Save plot
                plot_path = self.plots_dir / f"distribution_{classifier_name}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                plot_paths[f"distribution_{classifier_name}"] = str(plot_path)
            
            return plot_paths
            
        except Exception as e:
            logger.error(f"Error creating classifier distribution plots: {e}")
            return {}
    
    def create_comparison_plots(self, results_df: pd.DataFrame, 
                              baseline_model: str = "base") -> Dict[str, str]:
        """Create comparison plots showing improvements and regressions."""
        logger.info("Creating comparison plots...")
        
        plot_paths = {}
        
        # Get delta columns
        delta_cols = [col for col in results_df.columns if col.startswith("delta_")]
        
        if not delta_cols:
            logger.warning("No delta columns found for comparison plots")
            return plot_paths
        
        # Create improvement distribution plots
        improvement_plots = self._create_improvement_distributions(results_df, delta_cols)
        plot_paths.update(improvement_plots)
        
        # Create improvement summary plot
        summary_plot = self._create_improvement_summary(results_df, delta_cols)
        if summary_plot:
            plot_paths["improvement_summary"] = summary_plot
        
        logger.info(f"Created {len(plot_paths)} comparison plots")
        return plot_paths
    
    def _create_improvement_distributions(self, results_df: pd.DataFrame, 
                                        delta_cols: List[str]) -> Dict[str, str]:
        """Create distribution plots for improvement scores."""
        plot_paths = {}
        
        try:
            # Group by classifier
            classifier_groups = {}
            for col in delta_cols:
                parts = col.split("_")
                if len(parts) >= 5:
                    classifier_name = parts[-2]  # delta_{model}_vs_{baseline}_{classifier}_score
                    if classifier_name not in classifier_groups:
                        classifier_groups[classifier_name] = []
                    classifier_groups[classifier_name].append(col)
            
            # Create plots for each classifier
            for classifier_name, cols in classifier_groups.items():
                if len(cols) < 2:
                    continue
                
                # Create subplot grid
                n_cols = min(3, len(cols))
                n_rows = (len(cols) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
                if len(cols) == 1:
                    axes = [axes]
                elif n_rows == 1:
                    axes = axes.reshape(1, -1)
                else:
                    axes = axes.flatten()
                
                # Create plots for each model
                for i, col in enumerate(cols):
                    if i >= len(axes):
                        break
                        
                    ax = axes[i]
                    deltas = results_df[col].dropna()
                    
                    if len(deltas) > 0:
                        # Create histogram
                        ax.hist(deltas, bins=30, alpha=0.7, edgecolor='black')
                        
                        # Add zero line
                        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No Change')
                        
                        # Add statistics
                        mean_delta = deltas.mean()
                        improved_rate = (deltas > 0).mean()
                        ax.axvline(mean_delta, color='blue', linestyle='--', 
                                  label=f'Mean: {mean_delta:.3f}')
                        
                        model_name = col.split("_")[1]
                        ax.set_xlabel('Improvement Score (Positive = Better)')
                        ax.set_ylabel('Frequency')
                        ax.set_title(f'{model_name.title()} vs Baseline\n'
                                   f'Improved: {improved_rate:.1%}')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                
                # Hide empty subplots
                for i in range(len(cols), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                
                # Save plot
                plot_path = self.plots_dir / f"improvement_{classifier_name}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                plot_paths[f"improvement_{classifier_name}"] = str(plot_path)
            
            return plot_paths
            
        except Exception as e:
            logger.error(f"Error creating improvement distribution plots: {e}")
            return {}
    
    def _create_improvement_summary(self, results_df: pd.DataFrame, 
                                  delta_cols: List[str]) -> Optional[str]:
        """Create summary plot showing improvement statistics."""
        try:
            # Calculate summary statistics
            summary_data = []
            for col in delta_cols:
                deltas = results_df[col].dropna()
                if len(deltas) > 0:
                    parts = col.split("_")
                    model_name = parts[1]
                    classifier_name = parts[-2]
                    
                    summary_data.append({
                        'Model': model_name,
                        'Classifier': classifier_name,
                        'Mean_Improvement': deltas.mean(),
                        'Std_Improvement': deltas.std(),
                        'Improved_Rate': (deltas > 0).mean(),
                        'Median_Improvement': deltas.median()
                    })
            
            if not summary_data:
                return None
            
            df_summary = pd.DataFrame(summary_data)
            
            # Create summary plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Mean improvement by model
            model_means = df_summary.groupby('Model')['Mean_Improvement'].mean()
            model_means.plot(kind='bar', ax=ax1, color='skyblue')
            ax1.set_title('Mean Improvement by Model')
            ax1.set_ylabel('Mean Improvement Score')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Improved rate by model
            model_rates = df_summary.groupby('Model')['Improved_Rate'].mean()
            model_rates.plot(kind='bar', ax=ax2, color='lightgreen')
            ax2.set_title('Improvement Rate by Model')
            ax2.set_ylabel('Fraction of Improved Samples')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Mean improvement by classifier
            classifier_means = df_summary.groupby('Classifier')['Mean_Improvement'].mean()
            classifier_means.plot(kind='bar', ax=ax3, color='salmon')
            ax3.set_title('Mean Improvement by Classifier')
            ax3.set_ylabel('Mean Improvement Score')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Scatter plot of mean vs improved rate
            ax4.scatter(df_summary['Mean_Improvement'], df_summary['Improved_Rate'], 
                       alpha=0.7, s=100)
            ax4.set_xlabel('Mean Improvement Score')
            ax4.set_ylabel('Improvement Rate')
            ax4.set_title('Mean Improvement vs Improvement Rate')
            ax4.grid(True, alpha=0.3)
            
            # Add model labels to scatter plot
            for _, row in df_summary.iterrows():
                ax4.annotate(f"{row['Model']}\n{row['Classifier']}", 
                           (row['Mean_Improvement'], row['Improved_Rate']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.plots_dir / "improvement_summary.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error creating improvement summary plot: {e}")
            return None
    
    def create_all_plots(self, results_df: pd.DataFrame, 
                        baseline_model: str = "base") -> Dict[str, str]:
        """Create all visualization plots."""
        logger.info("Creating comprehensive visualization suite...")
        
        all_plots = {}
        
        # Create scatter plots
        scatter_plots = self.create_scatter_plots(results_df, baseline_model)
        all_plots.update(scatter_plots)
        
        # Create distribution plots
        distribution_plots = self.create_distribution_plots(results_df)
        all_plots.update(distribution_plots)
        
        # Create comparison plots
        comparison_plots = self.create_comparison_plots(results_df, baseline_model)
        all_plots.update(comparison_plots)
        
        # Create summary report
        summary_report = self._create_summary_report(results_df, all_plots)
        if summary_report:
            all_plots["summary_report"] = summary_report
        
        logger.info(f"Created {len(all_plots)} total plots")
        return all_plots
    
    def _create_summary_report(self, results_df: pd.DataFrame, 
                             plot_paths: Dict[str, str]) -> Optional[str]:
        """Create a summary HTML report with all plots and statistics."""
        try:
            # Calculate basic statistics
            score_cols = [col for col in results_df.columns 
                         if col.endswith("_score") and "prompt_" not in col]
            
            stats_html = "<h2>Dataset Statistics</h2>\n"
            stats_html += f"<p><strong>Total prompts:</strong> {len(results_df)}</p>\n"
            stats_html += f"<p><strong>Models evaluated:</strong> {len([col for col in results_df.columns if col.startswith('output_') and not col.endswith('_score')])}</p>\n"
            stats_html += f"<p><strong>Classifiers used:</strong> {len(set([col.split('_')[-2] for col in score_cols if col.split('_')[-1] == 'score']))}</p>\n"
            
            # Add toxicity score statistics
            stats_html += "<h3>Toxicity Score Statistics</h3>\n"
            stats_html += "<table border='1' style='border-collapse: collapse; width: 100%;'>\n"
            stats_html += "<tr><th>Model</th><th>Classifier</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>\n"
            
            for col in score_cols:
                scores = results_df[col].dropna()
                if len(scores) > 0:
                    parts = col.split("_")
                    model_name = parts[1]
                    classifier_name = parts[-2]
                    
                    stats_html += f"<tr><td>{model_name}</td><td>{classifier_name}</td>"
                    stats_html += f"<td>{scores.mean():.4f}</td><td>{scores.std():.4f}</td>"
                    stats_html += f"<td>{scores.min():.4f}</td><td>{scores.max():.4f}</td></tr>\n"
            
            stats_html += "</table>\n"
            
            # Create HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Toxicity Evaluation Results</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #34495e; }}
                    h3 {{ color: #7f8c8d; }}
                    .plot-section {{ margin: 20px 0; }}
                    .plot-container {{ text-align: center; margin: 10px 0; }}
                    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                    table {{ margin: 10px 0; }}
                    th, td {{ padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>Toxicity Evaluation Results</h1>
                <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                {stats_html}
                
                <h2>Visualizations</h2>
            """
            
            # Add plots to HTML
            for plot_name, plot_path in plot_paths.items():
                if plot_path.endswith('.png'):
                    relative_path = Path(plot_path).relative_to(self.output_dir)
                    html_content += f"""
                    <div class="plot-section">
                        <h3>{plot_name.replace('_', ' ').title()}</h3>
                        <div class="plot-container">
                            <img src="{relative_path}" alt="{plot_name}">
                        </div>
                    </div>
                    """
            
            html_content += """
            </body>
            </html>
            """
            
            # Save HTML report
            report_path = self.output_dir / "evaluation_report.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error creating summary report: {e}")
            return None 