"""
Script to analyze and compare reward models trained with different seeds and model sizes.
Evaluates individual models and ensembles on toxicity datasets.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
from scipy import stats
from scipy.spatial.distance import cosine
import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from irl_utilities import RewardModel


class RewardModelAnalyzer:
    """Class to analyze and compare reward models trained with different seeds."""

    def __init__(self, 
                 model_specs: List[Dict[str, Any]], 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 output_dir: str = "analysis_results"):
        """
        Initialize the analyzer with model specifications.
        
        Args:
            model_specs: List of dictionaries with model specifications:
                         [{"size": "70m", "seed": 42, "checkpoint": 30, "hub_id": "..."}]
            device: Device to run models on
            output_dir: Directory to save analysis results
        """
        self.model_specs = model_specs
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load models
        self.models = {}
        self.tokenizers = {}
        self.load_models()
        
        # Store value head weights for analysis
        self.value_head_weights = {}
        self.extract_value_head_weights()
        
    def load_models(self):
        """Load all reward models based on specifications."""
        print("Loading reward models...")
        successful_loads = 0
        
        for spec in tqdm(self.model_specs):
            model_id = f"pythia-{spec['size']}-seed-{spec['seed']}"
            hub_id = spec.get('hub_id', f"ajagota71/toxicity-reward-model-v-head-max-margin-seed-{spec['seed']}-pythia-{spec['size']}-checkpoint-{spec['checkpoint']}")
            
            try:
                # Load tokenizer
                self.tokenizers[model_id] = AutoTokenizer.from_pretrained(hub_id)
                if self.tokenizers[model_id].pad_token is None:
                    self.tokenizers[model_id].pad_token = self.tokenizers[model_id].eos_token
                
                # Load model using the approach from the working notebook
                print(f"Loading model from {hub_id}")
                
                # First, get the base model from the repo
                base_model = AutoModelForCausalLM.from_pretrained(hub_id)
                base_model_name = base_model.config._name_or_path
                print(f"Base model name: {base_model_name}")
                
                # Create reward model
                reward_model = RewardModel(
                    model_name=base_model_name,
                    device=self.device,
                    num_unfrozen_layers=0
                )
                
                # Load value head weights
                try:
                    from huggingface_hub import hf_hub_download
                    v_head_path = hf_hub_download(repo_id=hub_id, filename="v_head.pt")
                    print(f"Downloaded v_head.pt from {hub_id} to {v_head_path}")
                    
                    v_head_state = torch.load(v_head_path, map_location=self.device)
                    
                    # Check the structure of v_head_state
                    if isinstance(v_head_state, dict) and 'v_head' in v_head_state:
                        # If it's a dictionary with a 'v_head' key (from RewardModel.save())
                        reward_model.v_head.load_state_dict(v_head_state['v_head'])
                        print("Loaded v_head weights from dictionary")
                    elif isinstance(v_head_state, dict) and 'weight' in v_head_state:
                        # If it's just the state dict of the v_head module
                        reward_model.v_head.load_state_dict(v_head_state)
                        print("Loaded v_head weights directly from state dict")
                    else:
                        # Try to load as a tensor
                        try:
                            reward_model.v_head.weight.data = v_head_state
                            print("Loaded v_head weights as tensor")
                        except:
                            print("WARNING: Could not load v_head weights, using default initialization")
                    
                    # Print value head stats
                    v_head_weight = reward_model.v_head.weight.data.cpu().numpy()
                    print(f"Value head stats - Shape: {v_head_weight.shape}, Mean: {np.mean(v_head_weight):.6f}")
                    
                    # Store the model
                    self.models[model_id] = reward_model
                    successful_loads += 1
                    print(f"Successfully loaded {model_id}")
                    
                except Exception as e:
                    print(f"Error loading v_head weights: {e}")
                    continue
                
            except Exception as e:
                print(f"Error loading model {model_id}: {e}")
                continue
        
        if successful_loads == 0:
            print("\n" + "="*80)
            print("ERROR: No models were successfully loaded!")
            print("Please check the following:")
            print("1. Verify that the model paths are correct")
            print("2. Ensure you have access to the HuggingFace models")
            print("3. Try downloading the models locally first using:")
            print("   from huggingface_hub import snapshot_download")
            print("   snapshot_download(repo_id='ajagota71/toxicity-reward-model-v-head-max-margin-seed-42-pythia-70m-checkpoint-30')")
            print("="*80 + "\n")
        else:
            print(f"Successfully loaded {successful_loads} models")
    
    def extract_value_head_weights(self):
        """Extract value head weights from all models for analysis."""
        print("Extracting value head weights...")
        for model_id, model in self.models.items():
            # Get the value head weights
            weights = model.v_head.weight.data.cpu().numpy()
            self.value_head_weights[model_id] = weights
    
    def analyze_value_heads(self):
        """Analyze the value head weights across models."""
        print("Analyzing value head weights...")
        
        # Check if we have any models
        if not self.value_head_weights:
            print("No value head weights available for analysis. Skipping.")
            return None
        
        # Create a directory for value head analysis
        value_head_dir = os.path.join(self.output_dir, "value_head_analysis")
        os.makedirs(value_head_dir, exist_ok=True)
        
        # Calculate pairwise cosine similarities between value heads
        model_ids = list(self.value_head_weights.keys())
        n_models = len(model_ids)
        
        if n_models < 2:
            print("Need at least 2 models for similarity analysis. Skipping.")
            return None
        
        similarity_matrix = np.zeros((n_models, n_models))
        
        for i, model_id1 in enumerate(model_ids):
            for j, model_id2 in enumerate(model_ids):
                w1 = self.value_head_weights[model_id1].flatten()
                w2 = self.value_head_weights[model_id2].flatten()
                similarity = 1 - cosine(w1, w2)  # Convert distance to similarity
                similarity_matrix[i, j] = similarity
        
        # Plot similarity heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(similarity_matrix, annot=True, fmt=".3f", 
                    xticklabels=model_ids, yticklabels=model_ids, cmap="viridis")
        plt.title("Cosine Similarity Between Value Head Weights")
        plt.tight_layout()
        plt.savefig(os.path.join(value_head_dir, "value_head_similarity.png"), dpi=300)
        
        # Save similarity matrix
        similarity_df = pd.DataFrame(similarity_matrix, index=model_ids, columns=model_ids)
        similarity_df.to_csv(os.path.join(value_head_dir, "value_head_similarity.csv"))
        
        # Dimensionality reduction for visualization if we have enough models
        if n_models >= 5:
            try:
                # Prepare data for UMAP
                value_head_array = np.array([self.value_head_weights[mid].flatten() for mid in model_ids])
                
                # Apply UMAP
                reducer = umap.UMAP(n_components=2, random_state=42)
                embedding = reducer.fit_transform(value_head_array)
                
                # Create DataFrame for plotting
                embedding_df = pd.DataFrame({
                    'UMAP1': embedding[:, 0],
                    'UMAP2': embedding[:, 1],
                    'model_id': model_ids,
                    'size': [mid.split('-')[1] for mid in model_ids],
                    'seed': [mid.split('-')[-1] for mid in model_ids]
                })
                
                # Plot with plotly
                fig = px.scatter(
                    embedding_df, x='UMAP1', y='UMAP2', 
                    color='size', symbol='seed', text='model_id',
                    title='UMAP Projection of Value Head Weights',
                    labels={'UMAP1': 'UMAP Dimension 1', 'UMAP2': 'UMAP Dimension 2'},
                    height=600, width=800
                )
                fig.update_traces(marker=dict(size=12), textposition='top center')
                fig.write_html(os.path.join(value_head_dir, "value_head_umap.html"))
                fig.write_image(os.path.join(value_head_dir, "value_head_umap.png"), scale=2)
                
            except Exception as e:
                print(f"Error in UMAP visualization: {e}")
        
        return similarity_matrix
    
    def evaluate_on_dataset(self, dataset_path: str, dataset_name: str, batch_size: int = 16, 
                           max_samples: Optional[int] = None, text_key: str = "output"):
        """
        Evaluate all models on a dataset.
        
        Args:
            dataset_path: Path or HuggingFace ID of the dataset
            dataset_name: Name for the dataset in results
            batch_size: Batch size for processing
            max_samples: Maximum number of samples to process (None for all)
            text_key: Key in the dataset that contains the text to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Evaluating models on {dataset_name} dataset...")
        
        # Check if we have any models
        if not self.models:
            print("No models available for evaluation. Skipping.")
            return None
        
        # Load dataset
        try:
            if '/' in dataset_path and not os.path.exists(dataset_path):
                # HuggingFace dataset
                dataset = load_dataset(dataset_path)
                if isinstance(dataset, dict) and 'train' in dataset:
                    dataset = dataset['train']
                data = dataset.to_pandas().to_dict('records')
            else:
                # Local file
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
        except Exception as e:
            print(f"Error loading dataset {dataset_path}: {e}")
            return None
        
        # Limit samples if specified
        if max_samples is not None and max_samples < len(data):
            data = data[:max_samples]
        
        print(f"Processing {len(data)} samples from {dataset_name} dataset")
        
        # Store results for each model
        results = {model_id: [] for model_id in self.models}
        
        # If no models were loaded, return empty results
        if not results:
            print("No models available for evaluation. Skipping.")
            return None
        
        # Process in batches
        for i in tqdm(range(0, len(data), batch_size), desc=f"Evaluating {dataset_name}"):
            batch = data[i:i+batch_size]
            texts = [item[text_key] if text_key in item else item['text'] for item in batch]
            
            # Process with each model
            for model_id, model in self.models.items():
                tokenizer = self.tokenizers[model_id]
                
                # Tokenize
                inputs = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get rewards
                with torch.no_grad():
                    rewards = model(**inputs)
                
                # Convert to list of floats
                rewards_list = rewards.squeeze().cpu().tolist()
                
                # Handle single item case
                if not isinstance(rewards_list, list):
                    rewards_list = [rewards_list]
                
                # Store results
                results[model_id].extend(rewards_list)
        
        # Create a directory for dataset evaluation
        dataset_dir = os.path.join(self.output_dir, f"{dataset_name}_evaluation")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save raw scores
        scores_df = pd.DataFrame(results)
        scores_df.to_csv(os.path.join(dataset_dir, "model_scores.csv"), index=False)
        
        # Calculate statistics
        stats_dict = {}
        for model_id, scores in results.items():
            stats_dict[model_id] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "median": np.median(scores),
                "q1": np.percentile(scores, 25),
                "q3": np.percentile(scores, 75)
            }
        
        # Save statistics
        stats_df = pd.DataFrame(stats_dict).T
        stats_df.to_csv(os.path.join(dataset_dir, "score_statistics.csv"))
        
        # Plot score distributions
        plt.figure(figsize=(15, 10))
        for model_id, scores in results.items():
            sns.kdeplot(scores, label=model_id)
        plt.title(f"Score Distribution on {dataset_name} Dataset")
        plt.xlabel("Reward Score")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(dataset_dir, "score_distributions.png"), dpi=300)
        
        # Create boxplot
        plt.figure(figsize=(15, 10))
        plt.boxplot([results[model_id] for model_id in results], labels=list(results.keys()))
        plt.title(f"Score Distribution on {dataset_name} Dataset")
        plt.xlabel("Model")
        plt.ylabel("Reward Score")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(dataset_dir, "score_boxplots.png"), dpi=300)
        
        return results
    
    def compare_paired_datasets(self, dataset1_path: str, dataset2_path: str, 
                               dataset1_name: str = "dataset1", dataset2_name: str = "dataset2",
                               batch_size: int = 16, max_samples: Optional[int] = None,
                               text_key: str = "output"):
        """
        Compare model performance on two paired datasets.
        
        Args:
            dataset1_path: Path or HuggingFace ID of the first dataset
            dataset2_path: Path or HuggingFace ID of the second dataset
            dataset1_name: Name for the first dataset in results
            dataset2_name: Name for the second dataset in results
            batch_size: Batch size for processing
            max_samples: Maximum number of samples to process (None for all)
            text_key: Key in the dataset that contains the text to evaluate
            
        Returns:
            Dictionary with comparison results
        """
        print(f"Comparing models on {dataset1_name} vs {dataset2_name} datasets...")
        
        # Check if we have any models
        if not self.models:
            print("No models available for comparison. Skipping.")
            return None
        
        # Evaluate on both datasets
        results1 = self.evaluate_on_dataset(dataset1_path, dataset1_name, batch_size, max_samples, text_key)
        results2 = self.evaluate_on_dataset(dataset2_path, dataset2_name, batch_size, max_samples, text_key)
        
        # If either evaluation failed, return None
        if results1 is None or results2 is None:
            print("Evaluation failed on one or both datasets. Skipping comparison.")
            return None
        
        # Create a directory for comparison
        comparison_dir = os.path.join(self.output_dir, f"{dataset1_name}_vs_{dataset2_name}")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Calculate score differences
        diff_results = {}
        for model_id in self.models:
            if model_id in results1 and model_id in results2:
                # Ensure same length
                min_len = min(len(results1[model_id]), len(results2[model_id]))
                diff_results[model_id] = [results2[model_id][i] - results1[model_id][i] for i in range(min_len)]
        
        # Save differences
        diff_df = pd.DataFrame(diff_results)
        diff_df.to_csv(os.path.join(comparison_dir, "score_differences.csv"), index=False)
        
        # Calculate statistics on differences
        diff_stats = {}
        for model_id, diffs in diff_results.items():
            diff_stats[model_id] = {
                "mean_diff": np.mean(diffs),
                "std_diff": np.std(diffs),
                "median_diff": np.median(diffs),
                "positive_diff_pct": np.mean([d > 0 for d in diffs]) * 100,  # % where dataset2 > dataset1
                "effect_size": np.mean(diffs) / np.std(diffs) if np.std(diffs) > 0 else 0  # Cohen's d
            }
        
        # Save difference statistics
        diff_stats_df = pd.DataFrame(diff_stats).T
        diff_stats_df.to_csv(os.path.join(comparison_dir, "difference_statistics.csv"))
        
        # Plot mean differences
        plt.figure(figsize=(12, 8))
        model_ids = list(diff_stats.keys())
        mean_diffs = [diff_stats[mid]["mean_diff"] for mid in model_ids]
        
        # Sort by mean difference
        sorted_indices = np.argsort(mean_diffs)
        sorted_model_ids = [model_ids[i] for i in sorted_indices]
        sorted_mean_diffs = [mean_diffs[i] for i in sorted_indices]
        
        plt.barh(sorted_model_ids, sorted_mean_diffs)
        plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
        plt.title(f"Mean Score Difference ({dataset2_name} - {dataset1_name})")
        plt.xlabel("Mean Difference")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, "mean_differences.png"), dpi=300)
        
        # Plot paired distributions for each model
        for model_id in self.models:
            if model_id in results1 and model_id in results2:
                plt.figure(figsize=(10, 6))
                plt.hist(results1[model_id], alpha=0.5, bins=30, label=dataset1_name)
                plt.hist(results2[model_id], alpha=0.5, bins=30, label=dataset2_name)
                plt.title(f"Score Distribution for {model_id}")
                plt.xlabel("Reward Score")
                plt.ylabel("Frequency")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(comparison_dir, f"{model_id}_distributions.png"), dpi=300)
        
        return {
            "dataset1_results": results1,
            "dataset2_results": results2,
            "diff_results": diff_results,
            "diff_stats": diff_stats
        }
    
    def analyze_ensemble_methods(self, dataset1_path: str, dataset2_path: str,
                                dataset1_name: str, dataset2_name: str,
                                batch_size: int = 16, max_samples: Optional[int] = None,
                                text_key: str = "output"):
        """
        Analyze different ensemble methods for combining model predictions.
        
        Args:
            dataset1_path: Path or HuggingFace ID of the first dataset (e.g., toxic)
            dataset2_path: Path or HuggingFace ID of the second dataset (e.g., detoxified)
            dataset1_name: Name for the first dataset
            dataset2_name: Name for the second dataset
            batch_size: Batch size for processing
            max_samples: Maximum number of samples to process (None for all)
            text_key: Key in the dataset that contains the text to evaluate
            
        Returns:
            Dictionary with ensemble analysis results
        """
        print("Analyzing ensemble methods...")
        
        # Get results from paired datasets
        comparison_results = self.compare_paired_datasets(
            dataset1_path, dataset2_path, 
            dataset1_name, dataset2_name,
            batch_size, max_samples, text_key
        )
        
        if comparison_results is None:
            print("Error comparing datasets")
            return None
        
        results1 = comparison_results["dataset1_results"]  # e.g., toxic
        results2 = comparison_results["dataset2_results"]  # e.g., detoxified
        
        # Create a directory for ensemble analysis
        ensemble_dir = os.path.join(self.output_dir, "ensemble_analysis")
        os.makedirs(ensemble_dir, exist_ok=True)
        
        # Prepare data for ensemble analysis
        model_ids = list(self.models.keys())
        
        # Create ground truth labels: 1 for dataset1 (toxic), 0 for dataset2 (detoxified)
        min_len = min(len(results1[model_ids[0]]), len(results2[model_ids[0]]))
        ground_truth = [1] * min_len + [0] * min_len
        
        # Combine scores from both datasets
        all_scores = {}
        for model_id in model_ids:
            all_scores[model_id] = results1[model_id][:min_len] + results2[model_id][:min_len]
        
        # Define ensemble methods
        ensemble_methods = {
            "mean": lambda scores: np.mean(scores, axis=0),
            "median": lambda scores: np.median(scores, axis=0),
            "max": lambda scores: np.max(scores, axis=0),
            "min": lambda scores: np.min(scores, axis=0),
            "weighted_mean": lambda scores: np.average(scores, axis=0, weights=weights)
        }
        
        # Calculate weights based on individual model performance
        # Higher AUC-ROC gets higher weight
        weights = []
        for model_id in model_ids:
            # Invert scores for ROC calculation since higher score should mean less toxic
            auc_score = roc_auc_score(ground_truth, [-s for s in all_scores[model_id]])
            weights.append(auc_score)
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Apply ensemble methods
        ensemble_scores = {}
        for method_name, method_fn in ensemble_methods.items():
            scores_array = np.array([all_scores[mid] for mid in model_ids])
            ensemble_scores[method_name] = method_fn(scores_array).tolist()
        
        # Evaluate ensemble methods
        ensemble_metrics = {}
        for method_name, scores in ensemble_scores.items():
            # Calculate threshold (mean of scores)
            threshold = np.mean(scores)
            
            # Convert to binary predictions (higher score = less toxic = 0)
            predictions = (np.array(scores) > threshold).astype(int)
            predictions = 1 - predictions  # Invert to match ground truth (1=toxic)
            
            # Calculate metrics
            accuracy = accuracy_score(ground_truth, predictions)
            f1 = f1_score(ground_truth, predictions)
            auc_roc = roc_auc_score(ground_truth, [-s for s in scores])  # Invert for ROC
            
            # Calculate precision-recall curve and AUC
            precision, recall, _ = precision_recall_curve(ground_truth, [-s for s in scores])
            pr_auc = auc(recall, precision)
            
            ensemble_metrics[method_name] = {
                "accuracy": accuracy,
                "f1": f1,
                "auc_roc": auc_roc,
                "pr_auc": pr_auc
            }
        
        # Add individual model metrics for comparison
        for model_id in model_ids:
            scores = all_scores[model_id]
            threshold = np.mean(scores)
            
            predictions = (np.array(scores) > threshold).astype(int)
            predictions = 1 - predictions
            
            accuracy = accuracy_score(ground_truth, predictions)
            f1 = f1_score(ground_truth, predictions)
            auc_roc = roc_auc_score(ground_truth, [-s for s in scores])
            
            precision, recall, _ = precision_recall_curve(ground_truth, [-s for s in scores])
            pr_auc = auc(recall, precision)
            
            ensemble_metrics[model_id] = {
                "accuracy": accuracy,
                "f1": f1,
                "auc_roc": auc_roc,
                "pr_auc": pr_auc
            }
        
        # Save ensemble metrics
        ensemble_df = pd.DataFrame(ensemble_metrics).T
        ensemble_df.to_csv(os.path.join(ensemble_dir, "ensemble_metrics.csv"))
        
        # Plot metrics comparison
        metrics_to_plot = ["accuracy", "f1", "auc_roc", "pr_auc"]
        
        fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 4*len(metrics_to_plot)))
        
        for i, metric in enumerate(metrics_to_plot):
            # Sort by metric value
            sorted_items = sorted(ensemble_metrics.items(), key=lambda x: x[1][metric], reverse=True)
            methods = [item[0] for item in sorted_items]
            values = [item[1][metric] for item in sorted_items]
            
            # Color ensemble methods differently
            colors = ['blue' if method in ensemble_methods else 'green' for method in methods]
            
            axes[i].barh(methods, values, color=colors)
            axes[i].set_title(f"{metric.upper()}")
            axes[i].set_xlim(0, 1)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(ensemble_dir, "ensemble_comparison.png"), dpi=300)
        
        # Create interactive plotly visualization
        fig = make_subplots(rows=len(metrics_to_plot), cols=1, 
                           subplot_titles=[m.upper() for m in metrics_to_plot],
                           vertical_spacing=0.1)
        
        for i, metric in enumerate(metrics_to_plot):
            sorted_items = sorted(ensemble_metrics.items(), key=lambda x: x[1][metric], reverse=True)
            methods = [item[0] for item in sorted_items]
            values = [item[1][metric] for item in sorted_items]
            
            colors = ['rgba(0, 0, 255, 0.7)' if method in ensemble_methods else 'rgba(0, 128, 0, 0.7)' 
                     for method in methods]
            
            fig.add_trace(
                go.Bar(
                    y=methods,
                    x=values,
                    orientation='h',
                    marker_color=colors,
                    text=[f"{v:.3f}" for v in values],
                    textposition='auto',
                    name=metric
                ),
                row=i+1, col=1
            )
            
            fig.update_xaxes(range=[0, 1], row=i+1, col=1)
        
        fig.update_layout(
            height=300*len(metrics_to_plot),
            width=900,
            title_text="Ensemble Methods vs Individual Models",
            showlegend=False
        )
        
        fig.write_html(os.path.join(ensemble_dir, "ensemble_comparison.html"))
        
        # Analyze pairwise correlations between model predictions
        correlation_matrix = np.zeros((len(model_ids), len(model_ids)))
        
        for i, model_id1 in enumerate(model_ids):
            for j, model_id2 in enumerate(model_ids):
                correlation = np.corrcoef(all_scores[model_id1], all_scores[model_id2])[0, 1]
                correlation_matrix[i, j] = correlation
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt=".3f", 
                   xticklabels=model_ids, yticklabels=model_ids, cmap="viridis")
        plt.title("Correlation Between Model Predictions")
        plt.tight_layout()
        plt.savefig(os.path.join(ensemble_dir, "model_correlation.png"), dpi=300)
        
        # Save correlation matrix
        corr_df = pd.DataFrame(correlation_matrix, index=model_ids, columns=model_ids)
        corr_df.to_csv(os.path.join(ensemble_dir, "model_correlation.csv"))
        
        return {
            "ensemble_scores": ensemble_scores,
            "ensemble_metrics": ensemble_metrics,
            "correlation_matrix": correlation_matrix,
            "weights": dict(zip(model_ids, weights))
        }
    
    def analyze_by_model_size(self, results_dict):
        """
        Analyze performance by model size.
        
        Args:
            results_dict: Dictionary with evaluation results
            
        Returns:
            DataFrame with size-based analysis
        """
        # Extract model sizes
        size_metrics = {}
        
        for model_id, metrics in results_dict.items():
            size = model_id.split('-')[1]  # Extract size from model_id
            
            if size not in size_metrics:
                size_metrics[size] = []
            
            size_metrics[size].append(metrics)
        
        # Calculate average metrics by size
        size_avg_metrics = {}
        for size, metrics_list in size_metrics.items():
            # Calculate average of each metric
            avg_metrics = {}
            for metric in metrics_list[0].keys():
                avg_metrics[metric] = np.mean([m[metric] for m in metrics_list])
            
            size_avg_metrics[size] = avg_metrics
        
        return pd.DataFrame(size_avg_metrics).T
    
    def analyze_by_seed(self, results_dict):
        """
        Analyze performance by seed.
        
        Args:
            results_dict: Dictionary with evaluation results
            
        Returns:
            DataFrame with seed-based analysis
        """
        # Extract seeds
        seed_metrics = {}
        
        for model_id, metrics in results_dict.items():
            seed = model_id.split('-')[-1]  # Extract seed from model_id
            
            if seed not in seed_metrics:
                seed_metrics[seed] = []
            
            seed_metrics[seed].append(metrics)
        
        # Calculate average metrics by seed
        seed_avg_metrics = {}
        for seed, metrics_list in seed_metrics.items():
            # Calculate average of each metric
            avg_metrics = {}
            for metric in metrics_list[0].keys():
                avg_metrics[metric] = np.mean([m[metric] for m in metrics_list])
            
            seed_avg_metrics[seed] = avg_metrics
        
        return pd.DataFrame(seed_avg_metrics).T
    
    def run_full_analysis(self, original_dataset_path, detoxified_dataset_path, 
                         toxicity_prompts_dataset_path=None, batch_size=16, max_samples=None):
        """
        Run a full analysis on all datasets.
        
        Args:
            original_dataset_path: Path to original (toxic) dataset
            detoxified_dataset_path: Path to detoxified dataset
            toxicity_prompts_dataset_path: Path to toxicity prompts dataset (optional)
            batch_size: Batch size for processing
            max_samples: Maximum number of samples to process (None for all)
            
        Returns:
            Dictionary with all analysis results
        """
        results = {}
        
        # Check if we have any models
        if not self.models:
            print("No models were loaded. Cannot perform analysis.")
            return results
        
        # Analyze value heads
        print("Analyzing value heads...")
        results["value_head_similarity"] = self.analyze_value_heads()
        
        # Compare original vs detoxified datasets
        print("Comparing original vs detoxified datasets...")
        results["original_vs_detoxified"] = self.compare_paired_datasets(
            original_dataset_path, detoxified_dataset_path,
            "original", "detoxified",
            batch_size=batch_size,
            max_samples=max_samples
        )
        
        # Analyze ensemble methods
        print("Analyzing ensemble methods...")
        results["ensemble_analysis"] = self.analyze_ensemble_methods(
            original_dataset_path, detoxified_dataset_path,
            "original", "detoxified",
            batch_size=batch_size,
            max_samples=max_samples
        )
        
        # If toxicity prompts dataset is provided, evaluate on it
        if toxicity_prompts_dataset_path:
            print("Evaluating on toxicity prompts dataset...")
            results["toxicity_prompts"] = self.evaluate_on_dataset(
                toxicity_prompts_dataset_path,
                "toxicity_prompts",
                batch_size=batch_size,
                max_samples=max_samples,
                text_key="text"  # Adjust based on the dataset structure
            )
        
        # Create summary report
        self.create_summary_report(results)
        
        return results
    
    def create_summary_report(self, results):
        """Create a summary report of all analyses."""
        print("Creating summary report...")
        
        # Create a directory for the summary
        summary_dir = os.path.join(self.output_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        
        # Extract key metrics
        summary = {
            "value_head_analysis": {
                "average_similarity": np.mean(results["value_head_similarity"]) if "value_head_similarity" in results else None
            },
            "original_vs_detoxified": {}
        }
        
        # Extract ensemble performance metrics
        if "ensemble_analysis" in results and results["ensemble_analysis"]:
            ensemble_metrics = results["ensemble_analysis"].get("ensemble_metrics", {})
            
            # Find best ensemble method
            if ensemble_metrics:
                best_method = max(ensemble_metrics.keys(), key=lambda k: ensemble_metrics[k]["auc_roc"] 
                                 if k in ensemble_methods else 0)
                
                summary["best_ensemble"] = {
                    "method": best_method,
                    "metrics": ensemble_metrics[best_method]
                }
                
                # Find best individual model
                individual_models = {k: v for k, v in ensemble_metrics.items() if k not in ensemble_methods}
                if individual_models:
                    best_model = max(individual_models.keys(), key=lambda k: individual_models[k]["auc_roc"])
                    
                    summary["best_individual"] = {
                        "model": best_model,
                        "metrics": individual_models[best_model]
                    }
                    
                    # Calculate improvement
                    if "best_ensemble" in summary and "best_individual" in summary:
                        best_ensemble = summary["best_ensemble"]["metrics"]
                        best_individual = summary["best_individual"]["metrics"]
                        
                        summary["ensemble_improvement"] = {
                            "accuracy": best_ensemble["accuracy"] - best_individual["accuracy"],
                            "f1": best_ensemble["f1"] - best_individual["f1"],
                            "auc_roc": best_ensemble["auc_roc"] - best_individual["auc_roc"],
                            "pr_auc": best_ensemble["pr_auc"] - best_individual["pr_auc"]
                        }
        
        # Extract correlation metrics
        if "ensemble_analysis" in results and results["ensemble_analysis"]:
            if "correlation_matrix" in results["ensemble_analysis"]:
                corr_matrix = results["ensemble_analysis"]["correlation_matrix"]
                # Calculate average correlation between models
                n_models = corr_matrix.shape[0]
                if n_models > 1:
                    # Sum all correlations and subtract diagonal (which are all 1)
                    total_corr = np.sum(corr_matrix) - n_models
                    # Divide by number of off-diagonal elements
                    avg_corr = total_corr / (n_models * (n_models - 1))
                    summary["average_model_correlation"] = float(avg_corr)
        
        # Save summary
        with open(os.path.join(summary_dir, "analysis_summary.json"), "w") as f:
            # Convert numpy values to Python types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.generic):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(i) for i in obj]
                else:
                    return obj
            
            json.dump(convert_to_serializable(summary), f, indent=2)
        
        # Create a visual summary
        self.create_visual_summary(summary, summary_dir)
        
        return summary
    
    def create_visual_summary(self, summary, output_dir):
        """Create visual summary of the analysis."""
        # Create a dashboard-like visualization
        try:
            # Create a multi-panel figure
            fig = plt.figure(figsize=(15, 12))
            fig.suptitle(f"Reward Model Ensemble Analysis Summary", fontsize=16)
            
            # Define grid layout
            gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
            
            # Panel 1: Ensemble vs Individual Performance
            if "best_ensemble" in summary and "best_individual" in summary:
                ax1 = fig.add_subplot(gs[0, 0])
                
                metrics = ["accuracy", "f1", "auc_roc", "pr_auc"]
                ensemble_values = [summary["best_ensemble"]["metrics"][m] for m in metrics]
                individual_values = [summary["best_individual"]["metrics"][m] for m in metrics]
                
                x = np.arange(len(metrics))
                width = 0.35
                
                ax1.bar(x - width/2, ensemble_values, width, label=f'Best Ensemble ({summary["best_ensemble"]["method"]})')
                ax1.bar(x + width/2, individual_values, width, label=f'Best Individual ({summary["best_individual"]["model"]})')
                
                ax1.set_title("Performance Comparison")
                ax1.set_xticks(x)
                ax1.set_xticklabels([m.upper() for m in metrics])
                ax1.set_ylim(0, 1)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Panel 2: Ensemble Improvement
            if "ensemble_improvement" in summary:
                ax2 = fig.add_subplot(gs[0, 1])
                
                metrics = ["accuracy", "f1", "auc_roc", "pr_auc"]
                improvement_values = [summary["ensemble_improvement"][m] for m in metrics]
                
                colors = ['green' if v > 0 else 'red' for v in improvement_values]
                
                ax2.bar(metrics, improvement_values, color=colors)
                ax2.set_title("Ensemble Improvement")
                ax2.set_xticklabels([m.upper() for m in metrics])
                ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax2.grid(True, alpha=0.3)
                
                # Add percentage labels
                for i, v in enumerate(improvement_values):
                    if v > 0:
                        ax2.text(i, v + 0.01, f"+{v*100:.1f}%", ha='center')
                    else:
                        ax2.text(i, v - 0.02, f"{v*100:.1f}%", ha='center')
            
            # Panel 3: Model Correlation Heatmap
            if "average_model_correlation" in summary:
                ax3 = fig.add_subplot(gs[1, 0])
                ax3.text(0.5, 0.5, f"Average Model Correlation: {summary['average_model_correlation']:.4f}", 
                        ha='center', va='center', fontsize=14)
                ax3.axis('off')
            
            # Panel 4: Value Head Similarity
            if "value_head_analysis" in summary and summary["value_head_analysis"]["average_similarity"] is not None:
                ax4 = fig.add_subplot(gs[1, 1])
                ax4.text(0.5, 0.5, f"Average Value Head Similarity: {summary['value_head_analysis']['average_similarity']:.4f}", 
                        ha='center', va='center', fontsize=14)
                ax4.axis('off')
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "analysis_summary.png"), dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error creating visual summary: {e}")


class RewardModelEnsemble:
    """Class to create and use an ensemble of reward models."""
    
    def __init__(self, 
                 model_specs: List[Dict[str, Any]], 
                 ensemble_method: str = "weighted_mean",
                 weights: Optional[Dict[str, float]] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the ensemble.
        
        Args:
            model_specs: List of dictionaries with model specifications:
                         [{"size": "70m", "seed": 42, "checkpoint": 30, "hub_id": "..."}]
            ensemble_method: Method to use for ensemble ("mean", "median", "max", "min", "weighted_mean")
            weights: Optional dictionary mapping model IDs to weights (for weighted_mean)
            device: Device to run models on
        """
        self.model_specs = model_specs
        self.ensemble_method = ensemble_method
        self.device = device
        
        # Load models
        self.models = {}
        self.tokenizers = {}
        self.analyzer = RewardModelAnalyzer(model_specs, device=device)
        
        # Use the analyzer to load models
        self.analyzer.load_models()
        self.models = self.analyzer.models
        self.tokenizers = self.analyzer.tokenizers
        
        # Set up weights
        self.weights = weights or {model_id: 1.0 for model_id in self.models}
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.values()}
        
        # Define ensemble methods
        self.ensemble_methods = {
            "mean": lambda scores: np.mean(scores, axis=0),
            "median": lambda scores: np.median(scores, axis=0),
            "max": lambda scores: np.max(scores, axis=0),
            "min": lambda scores: np.min(scores, axis=0),
            "weighted_mean": lambda scores: np.average(scores, axis=0, weights=list(self.weights.values()))
        }
    
    def predict(self, texts: List[str], batch_size: int = 16, max_length: int = 512) -> np.ndarray:
        """
        Generate predictions using the ensemble.
        
        Args:
            texts: List of texts to generate predictions for
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            
        Returns:
            Array of ensemble predictions
        """
        # Get predictions from each model
        all_predictions = {}
        
        for model_id, model in self.models.items():
            # Process in batches
            model_preds = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                inputs = self.tokenizers[model_id](
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get predictions
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Convert to list of floats
                batch_preds = outputs.squeeze().cpu().numpy()
                
                # Handle single item case
                if not isinstance(batch_preds, np.ndarray) or batch_preds.ndim == 0:
                    batch_preds = np.array([batch_preds])
                
                model_preds.extend(batch_preds)
            
            all_predictions[model_id] = np.array(model_preds)
        
        # Apply ensemble method
        predictions_array = np.array([all_predictions[model_id] for model_id in self.models])
        ensemble_preds = self.ensemble_methods[self.ensemble_method](predictions_array)
        
        return ensemble_preds
    
    def save(self, output_dir: str):
        """
        Save the ensemble configuration.
        
        Args:
            output_dir: Directory to save the ensemble configuration
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save configuration
        config = {
            "ensemble_method": self.ensemble_method,
            "weights": self.weights,
            "model_specs": self.model_specs
        }
        
        with open(os.path.join(output_dir, "ensemble_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Ensemble configuration saved to {output_dir}")
    
    @classmethod
    def load(cls, config_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Load an ensemble from a configuration file.
        
        Args:
            config_path: Path to the ensemble configuration file
            device: Device to run models on
            
        Returns:
            RewardModelEnsemble instance
        """
        with open(config_path, "r") as f:
            config = json.load(f)
        
        return cls(
            model_specs=config["model_specs"],
            ensemble_method=config["ensemble_method"],
            weights=config["weights"],
            device=device
        )


def download_models(model_specs):
    """
    Download models from HuggingFace Hub.
    
    Args:
        model_specs: List of dictionaries with model specifications
    """
    try:
        from huggingface_hub import snapshot_download
        
        print("Downloading models from HuggingFace Hub...")
        downloaded_paths = {}
        
        for spec in tqdm(model_specs):
            hub_id = spec.get('hub_id', f"ajagota71/toxicity-reward-model-v-head-max-margin-seed-{spec['seed']}-pythia-{spec['size']}-checkpoint-{spec['checkpoint']}")
            
            try:
                print(f"Downloading {hub_id}...")
                local_dir = f"models/{hub_id}"
                path = snapshot_download(repo_id=hub_id, local_dir=local_dir)
                downloaded_paths[hub_id] = path
                print(f"Successfully downloaded {hub_id} to {path}")
                
                # Check if v_head.pt exists in the downloaded directory
                v_head_path = os.path.join(path, "v_head.pt")
                if os.path.exists(v_head_path):
                    print(f"Found v_head.pt at {v_head_path}")
                else:
                    print(f"Warning: v_head.pt not found in {path}")
                    # List files in the directory
                    print("Files in the directory:")
                    for file in os.listdir(path):
                        print(f"  {file}")
            except Exception as e:
                print(f"Error downloading {hub_id}: {e}")
        
        return downloaded_paths
    except ImportError:
        print("huggingface_hub not installed. Please install with: pip install huggingface_hub")
        return {}


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description="Analyze and ensemble reward models")
    
    # Model specification options
    parser.add_argument("--model_sizes", nargs="+", default=["70m", "160m", "410m", "1b"],
                        help="Model sizes to analyze")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 200, 300, 400],
                        help="Seeds to analyze")
    parser.add_argument("--checkpoints", nargs="+", type=int, default=None,
                        help="Checkpoints to use (one per model size)")
    
    # Dataset options
    parser.add_argument("--original_dataset", type=str, default=None,
                        help="Path or HuggingFace ID for original (toxic) dataset")
    parser.add_argument("--detoxified_dataset", type=str, default=None,
                        help="Path or HuggingFace ID for detoxified dataset")
    parser.add_argument("--toxicity_prompts_dataset", type=str, default="allenai/real-toxicity-prompts",
                        help="Path or HuggingFace ID for toxicity prompts dataset")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for processing")
    
    # Analysis options
    parser.add_argument("--output_dir", default="reward_model_analysis",
                        help="Directory to save analysis results")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Only run analysis, don't create ensemble")
    parser.add_argument("--ensemble_method", default="weighted_mean",
                        choices=["mean", "median", "max", "min", "weighted_mean"],
                        help="Method to use for ensemble")
    parser.add_argument("--download_models", action="store_true",
                        help="Download models from HuggingFace Hub before analysis")
    
    args = parser.parse_args()
    
    # Create model specifications
    model_specs = []
    default_checkpoints = {
        "70m": 30,
        "160m": 50,
        "410m": 70,
        "1b": 70
    }
    
    # If checkpoints are provided, use them
    if args.checkpoints:
        if len(args.checkpoints) != len(args.model_sizes):
            print(f"Warning: Number of checkpoints ({len(args.checkpoints)}) doesn't match number of model sizes ({len(args.model_sizes)})")
            print("Using default checkpoints for missing values")
        
        checkpoints = {}
        for i, size in enumerate(args.model_sizes):
            if i < len(args.checkpoints):
                checkpoints[size] = args.checkpoints[i]
            else:
                checkpoints[size] = default_checkpoints.get(size, 30)
    else:
        checkpoints = default_checkpoints
    
    # Create model specs
    for size in args.model_sizes:
        for seed in args.seeds:
            model_specs.append({
                "size": size,
                "seed": seed,
                "checkpoint": checkpoints.get(size, 30),
                "hub_id": f"ajagota71/toxicity-reward-model-v-head-max-margin-seed-{seed}-pythia-{size}-checkpoint-{checkpoints.get(size, 30)}"
            })
    
    # Download models if requested
    downloaded_paths = {}
    if args.download_models:
        downloaded_paths = download_models(model_specs)
        
        # Update model specs with downloaded paths
        for spec in model_specs:
            hub_id = spec.get('hub_id')
            if hub_id in downloaded_paths:
                spec['local_path'] = downloaded_paths[hub_id]
    
    # Create analyzer
    analyzer = RewardModelAnalyzer(
        model_specs=model_specs,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=args.output_dir
    )
    
    # Run analysis
    results = analyzer.run_full_analysis(
        original_dataset_path=args.original_dataset or f"ajagota71/EleutherAI_pythia-{args.model_sizes[0]}_2000_samples_original",
        detoxified_dataset_path=args.detoxified_dataset or f"ajagota71/ajagota71_pythia-{args.model_sizes[0]}-detox-epoch-100_2000_samples_detoxified",
        toxicity_prompts_dataset_path=args.toxicity_prompts_dataset,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    # Create ensemble if requested
    if not args.analyze_only:
        print("\nCreating reward model ensemble...")
        
        # Get weights from analysis results
        weights = None
        if "ensemble_analysis" in results and "weights" in results["ensemble_analysis"]:
            weights = results["ensemble_analysis"]["weights"]
        
        # Create ensemble
        ensemble = RewardModelEnsemble(
            model_specs=model_specs,
            ensemble_method=args.ensemble_method,
            weights=weights
        )
        
        # Save ensemble
        ensemble_dir = os.path.join(args.output_dir, "ensemble")
        ensemble.save(ensemble_dir)
        
        print(f"Ensemble created and saved to {ensemble_dir}")


if __name__ == "__main__":
    main()