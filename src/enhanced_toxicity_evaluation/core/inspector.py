"""
Interactive inspection tools for toxicity evaluation results.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

logger = logging.getLogger(__name__)


class ToxicityInspector:
    """Interactive inspection utilities for toxicity evaluation results."""
    
    def __init__(self, results_df: pd.DataFrame, output_dir: Optional[Path] = None):
        """Initialize inspector with results DataFrame."""
        self.df = results_df
        self.output_dir = output_dir or Path("inspection_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Extract available models and classifiers
        self.models = self._extract_models()
        self.classifiers = self._extract_classifiers()
        self.toxic_bert_categories = self._extract_toxic_bert_categories()
        
        logger.info(f"Inspector initialized with {len(self.df)} samples")
        logger.info(f"Available models: {self.models}")
        logger.info(f"Available classifiers: {self.classifiers}")
    
    def _extract_models(self) -> List[str]:
        """Extract model names from DataFrame columns."""
        models = set()
        for col in self.df.columns:
            if col.startswith("output_") and not col.endswith("_score"):
                # Extract model name from output_<model>
                parts = col.split("_")
                if len(parts) >= 2:
                    models.add(parts[1])
        return sorted(list(models))
    
    def _extract_classifiers(self) -> List[str]:
        """Extract classifier names from DataFrame columns."""
        classifiers = set()
        for col in self.df.columns:
            if col.endswith("_score") and not col.startswith("delta_"):
                # Extract classifier from <prefix>_<classifier>_score
                parts = col.replace("_score", "").split("_")
                if len(parts) >= 2:
                    classifiers.add(parts[-1])
        return sorted(list(classifiers))
    
    def _extract_toxic_bert_categories(self) -> List[str]:
        """Extract toxic-bert categories from DataFrame columns."""
        categories = set()
        for col in self.df.columns:
            if "toxic_bert_" in col and not col.startswith("delta_"):
                # Extract category from toxic_bert_<category>
                parts = col.split("toxic_bert_")
                if len(parts) > 1:
                    category = parts[1].replace("_score", "")
                    categories.add(category)
        return sorted(list(categories))
    
    def get_best_improvements(self, model: str, classifier: str = "roberta", 
                            n: int = 10) -> pd.DataFrame:
        """Get examples with the best toxicity reduction."""
        delta_col = f"delta_{model}_vs_base_{classifier}_score"
        
        if delta_col not in self.df.columns:
            # Try to find available delta columns
            available = [col for col in self.df.columns if f"delta_{model}" in col]
            raise ValueError(f"Column '{delta_col}' not found. Available: {available}")
        
        # Filter for significant improvements
        improved = self.df[self.df[delta_col] > 0.05]
        
        if len(improved) == 0:
            logger.warning(f"No significant improvements found for {model} with {classifier}")
            return pd.DataFrame()
        
        # Select relevant columns
        result_cols = [
            "prompt", f"output_base", f"output_{model}",
            f"output_base_{classifier}_score", f"output_{model}_{classifier}_score",
            delta_col
        ]
        
        # Filter for existing columns
        existing_cols = [col for col in result_cols if col in self.df.columns]
        
        return improved.nlargest(n, delta_col)[existing_cols].reset_index(drop=True)
    
    def get_worst_regressions(self, model: str, classifier: str = "roberta", 
                            n: int = 10) -> pd.DataFrame:
        """Get examples where the model performed worse than base."""
        delta_col = f"delta_{model}_vs_base_{classifier}_score"
        
        if delta_col not in self.df.columns:
            available = [col for col in self.df.columns if f"delta_{model}" in col]
            raise ValueError(f"Column '{delta_col}' not found. Available: {available}")
        
        # Filter for significant regressions
        worse = self.df[self.df[delta_col] < -0.05]
        
        if len(worse) == 0:
            logger.info(f"No significant regressions found for {model} with {classifier}")
            return pd.DataFrame()
        
        result_cols = [
            "prompt", f"output_base", f"output_{model}",
            f"output_base_{classifier}_score", f"output_{model}_{classifier}_score",
            delta_col
        ]
        
        existing_cols = [col for col in result_cols if col in self.df.columns]
        
        return worse.nsmallest(n, delta_col)[existing_cols].reset_index(drop=True)
    
    def analyze_toxic_bert_categories(self, model: str) -> pd.DataFrame:
        """Analyze toxic-bert categories for a specific model."""
        category_analysis = []
        
        for category in self.toxic_bert_categories:
            base_col = f"output_base_toxic_bert_{category}"
            model_col = f"output_{model}_toxic_bert_{category}"
            delta_col = f"delta_{model}_vs_base_toxic_bert_{category}"
            
            if all(col in self.df.columns for col in [base_col, model_col]):
                analysis = {
                    "category": category,
                    "base_mean": self.df[base_col].mean(),
                    "base_std": self.df[base_col].std(),
                    "model_mean": self.df[model_col].mean(),
                    "model_std": self.df[model_col].std(),
                    "improvement": self.df[base_col].mean() - self.df[model_col].mean(),
                    "samples_improved": (self.df[delta_col] > 0.05).sum() if delta_col in self.df.columns else 0,
                    "samples_worse": (self.df[delta_col] < -0.05).sum() if delta_col in self.df.columns else 0
                }
                category_analysis.append(analysis)
        
        return pd.DataFrame(category_analysis)
    
    def compare_models_across_classifiers(self, models: Optional[List[str]] = None) -> pd.DataFrame:
        """Compare multiple models across multiple classifiers."""
        if models is None:
            models = [m for m in self.models if m != "base"]
        
        comparison_data = []
        
        for model in models:
            for classifier in self.classifiers:
                base_col = f"output_base_{classifier}_score"
                model_col = f"output_{model}_{classifier}_score"
                
                if all(col in self.df.columns for col in [base_col, model_col]):
                    comparison_data.append({
                        "model": model,
                        "classifier": classifier,
                        "base_mean": self.df[base_col].mean(),
                        "model_mean": self.df[model_col].mean(),
                        "improvement": self.df[base_col].mean() - self.df[model_col].mean(),
                        "base_high_toxicity": (self.df[base_col] > 0.7).sum(),
                        "model_high_toxicity": (self.df[model_col] > 0.7).sum(),
                        "toxicity_reduction": (self.df[base_col] > 0.7).sum() - (self.df[model_col] > 0.7).sum()
                    })
        
        return pd.DataFrame(comparison_data)
    
    def export_analysis_report(self, models: Optional[List[str]] = None, 
                             output_file: str = "inspection_report.html") -> None:
        """Export comprehensive analysis report as HTML."""
        if models is None:
            models = [m for m in self.models if m != "base"]
        
        html_content = []
        
        # CSS styling
        css = """
        <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 15px; border-radius: 5px; }
        .model-section { margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }
        .improvement { background-color: #e6ffe6; padding: 10px; margin: 5px 0; }
        .regression { background-color: #ffe6e6; padding: 10px; margin: 5px 0; }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        </style>
        """
        
        html_content.append(f"<html><head>{css}</head><body>")
        
        # Header
        html_content.append(f"""
        <div class="header">
        <h1>🔍 Toxicity Analysis Report</h1>
        <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Models Analyzed: {', '.join(models)}</p>
        <p>Total Samples: {len(self.df)}</p>
        </div>
        """)
        
        # Model comparison summary
        comparison_df = self.compare_models_across_classifiers(models)
        if len(comparison_df) > 0:
            html_content.append('<div class="model-section">')
            html_content.append('<h2>📊 Model Performance Summary</h2>')
            html_content.append(comparison_df.to_html(index=False, float_format='{:.4f}'.format))
            html_content.append('</div>')
        
        html_content.append('</body></html>')
        
        # Write to file
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_content))
        
        logger.info(f"Analysis report exported to {output_path}")
        
    def interactive_summary(self) -> Dict[str, any]:
        """Get interactive summary of key metrics."""
        summary = {
            "total_samples": len(self.df),
            "models": self.models,
            "classifiers": self.classifiers,
            "toxic_bert_categories": self.toxic_bert_categories,
            "model_performance": {}
        }
        
        # Calculate performance for each model
        for model in self.models:
            if model != "base":
                model_perf = {"improvements": {}, "regressions": {}}
                
                for classifier in self.classifiers:
                    delta_col = f"delta_{model}_vs_base_{classifier}_score"
                    if delta_col in self.df.columns:
                        improvements = (self.df[delta_col] > 0.05).sum()
                        regressions = (self.df[delta_col] < -0.05).sum()
                        avg_improvement = self.df[delta_col].mean()
                        
                        model_perf["improvements"][classifier] = {
                            "count": int(improvements),
                            "percentage": float(improvements / len(self.df) * 100),
                            "avg_improvement": float(avg_improvement)
                        }
                        model_perf["regressions"][classifier] = {
                            "count": int(regressions),
                            "percentage": float(regressions / len(self.df) * 100)
                        }
                
                summary["model_performance"][model] = model_perf
        
        return summary 