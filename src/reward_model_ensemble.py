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
import requests
import wandb  # Add wandb import

import sys
# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from irl_utilities import RewardModel
from rlhf_utilities import build_dataset  # Import build_dataset for proper dataset splitting


class RewardModelAnalyzer:
    """Class to analyze and compare reward models trained with different seeds."""

    def __init__(self, 
                 model_specs: List[Dict[str, Any]], 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 output_dir: str = "analysis_results",
                 use_wandb: bool = False,
                 wandb_project: str = "reward-model-ensemble",
                 wandb_entity: Optional[str] = None):
        """
        Initialize the analyzer with model specifications.
        
        Args:
            model_specs: List of dictionaries with model specifications:
                         [{"size": "70m", "seed": 42, "checkpoint": 30, "hub_id": "..."}]
            device: Device to run models on
            output_dir: Directory to save analysis results
            use_wandb: Whether to use Weights & Biases for logging
            wandb_project: W&B project name
            wandb_entity: W&B entity name
        """
        self.model_specs = model_specs
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup wandb
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=wandb_project, entity=wandb_entity, 
                      name=f"reward-model-analysis-{len(model_specs)}-models",
                      config={"model_specs": model_specs})
        
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
    
    def log_to_wandb(self, data_dict, prefix=""):
        """Log metrics and images to wandb."""
        if not self.use_wandb:
            return
            
        # Log metrics
        metrics_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, (int, float)):
                metrics_dict[f"{prefix}/{key}"] = value
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        metrics_dict[f"{prefix}/{key}/{subkey}"] = subvalue
        
        if metrics_dict:
            wandb.log(metrics_dict)
        
        # Log images
        for key, value in data_dict.items():
            if isinstance(value, str) and value.endswith((".png", ".jpg", ".jpeg")):
                if os.path.exists(value):
                    wandb.log({f"{prefix}/{key}": wandb.Image(value)})
    
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
        heatmap_path = os.path.join(value_head_dir, "value_head_similarity.png")
        plt.savefig(heatmap_path, dpi=300)
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({"value_head/similarity_heatmap": wandb.Image(heatmap_path)})
        
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
                umap_html_path = os.path.join(value_head_dir, "value_head_umap.html")
                umap_png_path = os.path.join(value_head_dir, "value_head_umap.png")
                fig.write_html(umap_html_path)
                fig.write_image(umap_png_path, scale=2)
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({"value_head/umap_projection": wandb.Image(umap_png_path)})
                
            except Exception as e:
                print(f"Error in UMAP visualization: {e}")
        
        return similarity_matrix
    
    def create_distribution_plot(self, results, plot_name, title="Reward Distribution"):
        """
        Create distribution plots for model rewards.
        
        Args:
            results: Dictionary mapping model IDs to lists of reward scores
            plot_name: Name for the plot file
            title: Title for the plot
        """
        # Create directory for plots
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Ensure all scores are flattened to simple numbers
        for model_id in results:
            # Convert any nested lists to flat lists
            flat_scores = []
            for score in results[model_id]:
                if isinstance(score, (list, np.ndarray)):
                    if len(score) > 0:
                        flat_scores.append(float(score[0]))  # Take first element if it's a list
                else:
                    flat_scores.append(float(score))
            results[model_id] = flat_scores
        
        # Create DataFrame for easier plotting
        data = []
        for model_id, scores in results.items():
            model_data = pd.DataFrame({
                'model': model_id,
                'score': scores
            })
            data.append(model_data)
        
        if not data:
            print(f"Warning: No data available for {plot_name}")
            return
        
        df = pd.concat(data, ignore_index=True)
        
        # Calculate statistics
        stats = df.groupby('model')['score'].agg(['mean', 'std', 'min', 'max', 'median'])
        stats['q1'] = df.groupby('model')['score'].quantile(0.25)
        stats['q3'] = df.groupby('model')['score'].quantile(0.75)
        
        # Save statistics
        stats_path = os.path.join(plots_dir, f"{plot_name}_stats.csv")
        stats.to_csv(stats_path)
        
        # Log statistics to wandb
        if self.use_wandb:
            # Log mean scores for each model
            for model_id in results:
                wandb.log({f"{plot_name}/mean_score/{model_id}": stats.loc[model_id, 'mean']})
            
            # Create a wandb table for all statistics
            wandb_table = wandb.Table(dataframe=stats.reset_index())
            wandb.log({f"{plot_name}/statistics": wandb_table})
        
        # Create KDE plot
        plt.figure(figsize=(15, 10))
        for model_id in results:
            sns.kdeplot(results[model_id], label=model_id)
        plt.title(title)
        plt.xlabel("Reward Score")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        kde_path = os.path.join(plots_dir, f"{plot_name}_kde.png")
        plt.savefig(kde_path, dpi=300)
        plt.close()
        
        # Create boxplot
        plt.figure(figsize=(15, 10))
        plt.boxplot([results[model_id] for model_id in results], labels=list(results.keys()))
        plt.title(title)
        plt.xlabel("Model")
        plt.ylabel("Reward Score")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        boxplot_path = os.path.join(plots_dir, f"{plot_name}_boxplot.png")
        plt.savefig(boxplot_path, dpi=300)
        plt.close()
        
        # Create violin plot
        plt.figure(figsize=(15, 10))
        sns.violinplot(x='model', y='score', data=df)
        plt.title(title)
        plt.xlabel("Model")
        plt.ylabel("Reward Score")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        violin_path = os.path.join(plots_dir, f"{plot_name}_violin.png")
        plt.savefig(violin_path, dpi=300)
        plt.close()
        
        # Create interactive plotly histogram
        fig = px.histogram(df, x='score', color='model', marginal='box', 
                          title=title, opacity=0.7, barmode='overlay')
        fig.update_layout(
            xaxis_title="Reward Score",
            yaxis_title="Count",
            legend_title="Model"
        )
        histogram_html_path = os.path.join(plots_dir, f"{plot_name}_histogram.html")
        histogram_png_path = os.path.join(plots_dir, f"{plot_name}_histogram.png")
        fig.write_html(histogram_html_path)
        fig.write_image(histogram_png_path, scale=2)
        
        # Log plots to wandb
        if self.use_wandb:
            wandb.log({
                f"{plot_name}/kde_plot": wandb.Image(kde_path),
                f"{plot_name}/boxplot": wandb.Image(boxplot_path),
                f"{plot_name}/violin_plot": wandb.Image(violin_path),
                f"{plot_name}/histogram": wandb.Image(histogram_png_path)
            })
    
    def evaluate_on_dataset(self, dataset, dataset_name: str, batch_size: int = 16, 
                           max_samples: Optional[int] = None, text_key: str = "output",
                           split: Optional[str] = None):
        """
        Evaluate all models on a dataset.
        
        Args:
            dataset: Dataset to evaluate on (can be path, HuggingFace ID, or pre-loaded dataset)
            dataset_name: Name for the dataset in results
            batch_size: Batch size for processing
            max_samples: Maximum number of samples to process (None for all)
            text_key: Key in the dataset that contains the text to evaluate
            split: Dataset split to evaluate on (train, test, or None for all)
            
        Returns:
            Dictionary with evaluation results
        """
        split_suffix = f"_{split}" if split else ""
        print(f"Evaluating models on {dataset_name}{split_suffix} dataset...")
        
        # Check if we have any models
        if not self.models:
            print("No models available for evaluation. Skipping.")
            return None
        
        # Handle different dataset types
        if isinstance(dataset, str):
            # Load dataset if it's a path or HuggingFace ID
            try:
                if dataset.startswith(("http", "https", "s3://")):
                    # Download from URL
                    response = requests.get(dataset)
                    data = response.json()
                elif os.path.exists(dataset):
                    # Load from local file
                    with open(dataset, "r") as f:
                        data = json.load(f)
                else:
                    # Try to load from HuggingFace
                    ds = load_dataset(dataset)
                    if "train" in ds:
                        ds = ds["train"]
                    data = ds.to_pandas().to_dict(orient="records")
            except Exception as e:
                print(f"Error loading dataset {dataset}: {e}")
                return None
        elif hasattr(dataset, '__getitem__') and hasattr(dataset, '__len__'):
            # Dataset is already loaded - could be a HuggingFace dataset or list
            data = dataset
        else:
            print(f"Unsupported dataset type: {type(dataset)}")
            return None
        
        # Get dataset length based on type
        if hasattr(data, '__len__'):
            data_len = len(data)
        else:
            print("Cannot determine dataset length. Skipping.")
            return None
        
        # Limit samples if specified
        if max_samples is not None and max_samples < data_len:
            if hasattr(data, 'select'):
                # HuggingFace dataset
                data = data.select(range(max_samples))
                data_len = max_samples
            else:
                # List or other sequence
                data = data[:max_samples]
                data_len = max_samples
        
        print(f"Processing {data_len} samples from {dataset_name}{split_suffix} dataset")
        
        # Store results for each model
        results = {model_id: [] for model_id in self.models}
        
        # Store the texts for reference
        all_texts = []
        
        # Store text IDs if available
        text_ids = []
        
        # Store toxicity metrics if available (for toxicity prompts dataset)
        toxicity_metrics = {
            "toxicity": [],
            "severe_toxicity": [],
            "insult": [],
            "threat": [],
            "identity_attack": [],
            "profanity": [],
            "sexually_explicit": [],
            "flirtation": []
        }
        has_toxicity_metrics = False
        
        # If no models were loaded, return empty results
        if not results:
            print("No models available for evaluation. Skipping.")
            return None
        
        # Process in batches
        for i in tqdm(range(0, data_len, batch_size), desc=f"Evaluating {dataset_name}{split_suffix}"):
            # Get batch based on dataset type
            if hasattr(data, 'select'):
                # HuggingFace dataset
                end_idx = min(i + batch_size, data_len)
                batch = data.select(range(i, end_idx))
            else:
                # List or other sequence
                batch = data[i:i+batch_size]
            
            # Extract texts from batch based on dataset type
            if hasattr(batch, 'query'):
                # HuggingFace dataset with query field (from RLHF utilities)
                texts = batch['query']
                # Try to get IDs if available
                if hasattr(batch, 'id'):
                    batch_ids = batch['id']
                else:
                    batch_ids = [f"{dataset_name}_{i+j}" for j in range(len(texts))]
                
                # Check for toxicity metrics in the toxicity prompts dataset
                if dataset_name == "toxicity_prompts" and hasattr(batch, 'prompt'):
                    for metric in toxicity_metrics.keys():
                        if hasattr(batch.prompt, metric):
                            toxicity_metrics[metric].extend(batch.prompt[metric])
                            has_toxicity_metrics = True
                
            elif hasattr(batch, 'features') and 'query' in batch.features:
                # Another way HuggingFace might structure it
                texts = [item['query'] for item in batch]
                # Try to get IDs if available
                if 'id' in batch.features:
                    batch_ids = [item['id'] for item in batch]
                else:
                    batch_ids = [f"{dataset_name}_{i+j}" for j in range(len(texts))]
                
                # Check for toxicity metrics in the toxicity prompts dataset
                if dataset_name == "toxicity_prompts":
                    for j, item in enumerate(batch):
                        if 'prompt' in item and isinstance(item['prompt'], dict):
                            for metric in toxicity_metrics.keys():
                                if metric in item['prompt']:
                                    toxicity_metrics[metric].append(item['prompt'][metric])
                                    has_toxicity_metrics = True
                
            elif isinstance(batch, list) and isinstance(batch[0], dict):
                # List of dictionaries
                texts = []
                batch_ids = []
                for j, item in enumerate(batch):
                    # Get text
                    if "prompt" in item and isinstance(item["prompt"], dict) and "text" in item["prompt"]:
                        texts.append(item["prompt"]["text"])
                        
                        # Check for toxicity metrics in the toxicity prompts dataset
                        if dataset_name == "toxicity_prompts":
                            for metric in toxicity_metrics.keys():
                                if metric in item["prompt"]:
                                    toxicity_metrics[metric].append(item["prompt"][metric])
                                    has_toxicity_metrics = True
                                
                    elif text_key in item:
                        texts.append(item[text_key])
                    elif "text" in item:
                        texts.append(item["text"])
                    else:
                        # Use a default empty string if no text field is found
                        texts.append("")
                    
                    # Get ID
                    if "id" in item:
                        batch_ids.append(item["id"])
                    else:
                        batch_ids.append(f"{dataset_name}_{i+j}")
            else:
                # Try to convert to list if it's another iterable
                try:
                    texts = list(batch)
                    batch_ids = [f"{dataset_name}_{i+j}" for j in range(len(texts))]
                except:
                    print(f"Could not extract texts from batch of type {type(batch)}")
                    continue
            
            # Store texts and IDs
            all_texts.extend(texts)
            text_ids.extend(batch_ids)
            
            # Score texts with each model
            for model_id, model in self.models.items():
                tokenizer = self.tokenizers[model_id]
                
                # Tokenize texts
                inputs = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Get rewards
                with torch.no_grad():
                    rewards = model(**inputs)
                
                # Convert to list of floats
                rewards_list = rewards.cpu().numpy().flatten().tolist()
                
                # Add to results
                results[model_id].extend(rewards_list)
        
        # Create plots
        plot_name = f"{dataset_name}{split_suffix}_distribution"
        self.create_distribution_plot(results, plot_name, 
                                     title=f"Reward Distribution on {dataset_name}{split_suffix} Dataset")
        
        # Create a detailed table with all scores for each text
        if self.use_wandb and all_texts:
            # Create a DataFrame with text IDs, texts, and scores from each model
            table_data = {
                "text_id": text_ids,
                "text": [t[:1000] if len(t) > 1000 else t for t in all_texts]  # Truncate long texts
            }
            
            # Add toxicity metrics if available
            if has_toxicity_metrics and dataset_name == "toxicity_prompts":
                print(f"Adding toxicity metrics to the detailed scores table")
                for metric, values in toxicity_metrics.items():
                    if values:  # Only add if we have values
                        # Ensure we have the right number of values
                        if len(values) > len(all_texts):
                            values = values[:len(all_texts)]
                        elif len(values) < len(all_texts):
                            # Pad with NaN
                            values = values + [float('nan')] * (len(all_texts) - len(values))
                        
                        table_data[f"original_{metric}"] = values
            
            # Add scores from each model
            for model_id, scores in results.items():
                # Extract seed from model_id for cleaner column names
                seed = model_id.split('-')[-1]
                table_data[f"score_seed_{seed}"] = scores[:len(all_texts)]
            
            # Create DataFrame
            df = pd.DataFrame(table_data)
            
            # Log to wandb as a table
            wandb_table = wandb.Table(dataframe=df)
            wandb.log({f"detailed_scores/{dataset_name}{split_suffix}": wandb_table})
            
            # Also save locally
            scores_dir = os.path.join(self.output_dir, "detailed_scores")
            os.makedirs(scores_dir, exist_ok=True)
            df.to_csv(os.path.join(scores_dir, f"{dataset_name}{split_suffix}_detailed_scores.csv"), index=False)
            
            # Create correlation heatmap between different seeds
            score_columns = [col for col in df.columns if col.startswith("score_seed_")]
            
            # If we have toxicity metrics, include them in correlation analysis
            if has_toxicity_metrics:
                metric_columns = [col for col in df.columns if col.startswith("original_")]
                correlation_columns = score_columns + metric_columns
                
                # Create a correlation matrix between model scores and toxicity metrics
                if len(correlation_columns) > 1:
                    corr_matrix = df[correlation_columns].corr()
                    
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="viridis")
                    plt.title(f"Correlation Between Model Scores and Toxicity Metrics on {dataset_name}{split_suffix}")
                    plt.tight_layout()
                    
                    corr_path = os.path.join(scores_dir, f"{dataset_name}{split_suffix}_metrics_correlation.png")
                    plt.savefig(corr_path, dpi=300)
                    plt.close()
                    
                    # Log to wandb
                    wandb.log({f"metrics_correlation/{dataset_name}{split_suffix}": wandb.Image(corr_path)})
            
            # Create correlation heatmap between different seeds
            if len(score_columns) > 1:
                corr_matrix = df[score_columns].corr()
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="viridis")
                plt.title(f"Score Correlation Between Seeds on {dataset_name}{split_suffix}")
                plt.tight_layout()
                
                corr_path = os.path.join(scores_dir, f"{dataset_name}{split_suffix}_seed_correlation.png")
                plt.savefig(corr_path, dpi=300)
                plt.close()
                
                # Log to wandb
                wandb.log({f"seed_correlation/{dataset_name}{split_suffix}": wandb.Image(corr_path)})
        
        return results
    
    def compare_paired_datasets(self, dataset1, dataset2, 
                               dataset1_name: str = "dataset1", dataset2_name: str = "dataset2",
                               batch_size: int = 16, max_samples: Optional[int] = None,
                               text_key: str = "output", split: Optional[str] = None):
        """
        Compare model performance on two paired datasets.
        
        Args:
            dataset1: First dataset (path, HuggingFace ID, or pre-loaded dataset)
            dataset2: Second dataset (path, HuggingFace ID, or pre-loaded dataset)
            dataset1_name: Name for the first dataset in results
            dataset2_name: Name for the second dataset in results
            batch_size: Batch size for processing
            max_samples: Maximum number of samples to process (None for all)
            text_key: Key in the dataset that contains the text to evaluate
            split: Dataset split to evaluate on (train, test, or None for all)
            
        Returns:
            Dictionary with comparison results
        """
        split_suffix = f"_{split}" if split else ""
        print(f"Comparing models on {dataset1_name} vs {dataset2_name}{split_suffix} datasets...")
        
        # Check if we have any models
        if not self.models:
            print("No models available for comparison. Skipping.")
            return None
        
        # Evaluate on both datasets
        results1 = self.evaluate_on_dataset(dataset1, dataset1_name, batch_size, max_samples, text_key, split)
        results2 = self.evaluate_on_dataset(dataset2, dataset2_name, batch_size, max_samples, text_key, split)
        
        # If either evaluation failed, return None
        if results1 is None or results2 is None:
            print("Evaluation failed on one or both datasets. Skipping comparison.")
            return None
        
        # Create a directory for comparison
        comparison_dir = os.path.join(self.output_dir, f"{dataset1_name}_vs_{dataset2_name}{split_suffix}")
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
        diff_path = os.path.join(comparison_dir, "score_differences.csv")
        diff_df.to_csv(diff_path, index=False)
        
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
        diff_stats_path = os.path.join(comparison_dir, "difference_statistics.csv")
        diff_stats_df.to_csv(diff_stats_path)
        
        # Log to wandb
        if self.use_wandb:
            # Log mean differences for each model
            for model_id, stats in diff_stats.items():
                wandb.log({f"comparison{split_suffix}/{dataset1_name}_vs_{dataset2_name}/mean_diff/{model_id}": stats["mean_diff"]})
            
            # Create a wandb table for all statistics
            wandb_table = wandb.Table(dataframe=diff_stats_df.reset_index())
            wandb.log({f"comparison{split_suffix}/{dataset1_name}_vs_{dataset2_name}/statistics": wandb_table})
        
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
        plt.title(f"Mean Score Difference ({dataset2_name} - {dataset1_name}){split_suffix}")
        plt.xlabel("Mean Difference")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        mean_diff_path = os.path.join(comparison_dir, "mean_differences.png")
        plt.savefig(mean_diff_path, dpi=300)
        plt.close()
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({f"comparison{split_suffix}/{dataset1_name}_vs_{dataset2_name}/mean_differences": wandb.Image(mean_diff_path)})
        
        # Plot paired distributions for each model
        for model_id in self.models:
            if model_id in results1 and model_id in results2:
                plt.figure(figsize=(10, 6))
                plt.hist(results1[model_id], alpha=0.5, bins=30, label=dataset1_name)
                plt.hist(results2[model_id], alpha=0.5, bins=30, label=dataset2_name)
                plt.title(f"Score Distribution for {model_id}{split_suffix}")
                plt.xlabel("Reward Score")
                plt.ylabel("Frequency")
                plt.legend()
                plt.grid(True, alpha=0.3)
                dist_path = os.path.join(comparison_dir, f"{model_id}_distributions.png")
                plt.savefig(dist_path, dpi=300)
                plt.close()
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({f"comparison{split_suffix}/{dataset1_name}_vs_{dataset2_name}/distribution/{model_id}": wandb.Image(dist_path)})
        
        return {
            "dataset1_results": results1,
            "dataset2_results": results2,
            "diff_results": diff_results,
            "diff_stats": diff_stats
        }
    
    def analyze_ensemble_methods(self, dataset1, dataset2,
                                dataset1_name: str, dataset2_name: str,
                                batch_size: int = 16, max_samples: Optional[int] = None,
                                text_key: str = "output", split: Optional[str] = None):
        """
        Analyze different ensemble methods for combining model predictions.
        
        Args:
            dataset1: First dataset (e.g., toxic)
            dataset2: Second dataset (e.g., detoxified)
            dataset1_name: Name for the first dataset
            dataset2_name: Name for the second dataset
            batch_size: Batch size for processing
            max_samples: Maximum number of samples to process (None for all)
            text_key: Key in the dataset that contains the text to evaluate
            split: Dataset split to evaluate on (train, test, or None for all)
            
        Returns:
            Dictionary with ensemble analysis results
        """
        split_suffix = f"_{split}" if split else ""
        print(f"Analyzing ensemble methods on {split_suffix} split...")
        
        # Get results from paired datasets
        comparison_results = self.compare_paired_datasets(
            dataset1, dataset2, 
            dataset1_name, dataset2_name,
            batch_size, max_samples, text_key, split
        )
        
        if comparison_results is None:
            print("Error comparing datasets")
            return None
        
        results1 = comparison_results["dataset1_results"]  # e.g., toxic
        results2 = comparison_results["dataset2_results"]  # e.g., detoxified
        
        # Create a directory for ensemble analysis
        ensemble_dir = os.path.join(self.output_dir, f"ensemble_analysis{split_suffix}")
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
        ensemble_metrics_path = os.path.join(ensemble_dir, "ensemble_metrics.csv")
        ensemble_df.to_csv(ensemble_metrics_path)
        
        # Log to wandb
        if self.use_wandb:
            # Log metrics for each method
            for method_name, metrics in ensemble_metrics.items():
                for metric_name, value in metrics.items():
                    wandb.log({f"ensemble{split_suffix}/{method_name}/{metric_name}": value})
            
            # Create a wandb table for all metrics
            wandb_table = wandb.Table(dataframe=ensemble_df.reset_index().rename(columns={"index": "method"}))
            wandb.log({f"ensemble{split_suffix}/metrics": wandb_table})
        
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
        ensemble_comparison_path = os.path.join(ensemble_dir, "ensemble_comparison.png")
        plt.savefig(ensemble_comparison_path, dpi=300)
        plt.close()
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({f"ensemble{split_suffix}/comparison": wandb.Image(ensemble_comparison_path)})
        
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
            title_text=f"Ensemble Methods vs Individual Models{split_suffix}",
            showlegend=False
        )
        
        ensemble_html_path = os.path.join(ensemble_dir, "ensemble_comparison.html")
        fig.write_html(ensemble_html_path)
        
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
        plt.title(f"Correlation Between Model Predictions{split_suffix}")
        plt.tight_layout()
        correlation_path = os.path.join(ensemble_dir, "model_correlation.png")
        plt.savefig(correlation_path, dpi=300)
        plt.close()
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({f"ensemble{split_suffix}/model_correlation": wandb.Image(correlation_path)})
        
        # Save correlation matrix
        corr_df = pd.DataFrame(correlation_matrix, index=model_ids, columns=model_ids)
        corr_df.to_csv(os.path.join(ensemble_dir, "model_correlation.csv"))
        
        return {
            "ensemble_scores": ensemble_scores,
            "ensemble_metrics": ensemble_metrics,
            "correlation_matrix": correlation_matrix,
            "weights": dict(zip(model_ids, weights))
        }
    
    def run_full_analysis(self, original_dataset_path, detoxified_dataset_path, 
                         toxicity_prompts_dataset_path=None, batch_size=16, max_samples=None,
                         train_test_split=0.8, seed=42):
        """
        Run a full analysis on all datasets with proper train/test splitting.
        
        Args:
            original_dataset_path: Path to original (toxic) dataset
            detoxified_dataset_path: Path to detoxified dataset
            toxicity_prompts_dataset_path: Path to toxicity prompts dataset (optional)
            batch_size: Batch size for processing
            max_samples: Maximum number of samples to process (None for all)
            train_test_split: Fraction of data to use for training
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with all analysis results
        """
        results = {}
        
        # Check if we have any models
        if not self.models:
            print("No models were loaded. Cannot perform analysis.")
            return results
        
        # Analyze value heads
        print("Analyzing value head weights...")
        results["value_head_similarity"] = self.analyze_value_heads()
        
        # Load and split paired datasets
        print("Loading and splitting paired datasets...")
        paired_datasets = self.load_and_split_paired_datasets(
            original_dataset_path, detoxified_dataset_path,
            train_test_split=train_test_split, seed=seed
        )
        
        if paired_datasets:
            # Analyze train split
            print("Analyzing train split...")
            results["train"] = {
                "original_vs_detoxified": self.compare_paired_datasets(
                    paired_datasets['train']['original'], 
                    paired_datasets['train']['detoxified'],
                    "original", "detoxified",
                    batch_size=batch_size,
                    max_samples=max_samples,
                    split="train"
                ),
                "ensemble_analysis": self.analyze_ensemble_methods(
                    paired_datasets['train']['original'], 
                    paired_datasets['train']['detoxified'],
                    "original", "detoxified",
                    batch_size=batch_size,
                    max_samples=max_samples,
                    split="train"
                )
            }
            
            # Analyze test split
            print("Analyzing test split...")
            results["test"] = {
                "original_vs_detoxified": self.compare_paired_datasets(
                    paired_datasets['test']['original'], 
                    paired_datasets['test']['detoxified'],
                    "original", "detoxified",
                    batch_size=batch_size,
                    max_samples=max_samples,
                    split="test"
                ),
                "ensemble_analysis": self.analyze_ensemble_methods(
                    paired_datasets['test']['original'], 
                    paired_datasets['test']['detoxified'],
                    "original", "detoxified",
                    batch_size=batch_size,
                    max_samples=max_samples,
                    split="test"
                )
            }
        
        # If toxicity prompts dataset is provided, evaluate on it
        if toxicity_prompts_dataset_path:
            print("Loading and splitting toxicity prompts dataset...")
            toxicity_datasets = self.load_and_split_toxicity_prompts(
                dataset_path=toxicity_prompts_dataset_path,
                seed=seed
            )
            
            if toxicity_datasets:
                # Evaluate on train split
                print("Evaluating on toxicity prompts train split...")
                results["toxicity_prompts_train"] = self.evaluate_on_dataset(
                    toxicity_datasets['train'],
                    "toxicity_prompts",
                    batch_size=batch_size,
                    max_samples=max_samples,
                    text_key="query",
                    split="train"
                )
                
                # Evaluate on test split
                print("Evaluating on toxicity prompts test split...")
                results["toxicity_prompts_test"] = self.evaluate_on_dataset(
                    toxicity_datasets['test'],
                    "toxicity_prompts",
                    batch_size=batch_size,
                    max_samples=max_samples,
                    text_key="query",
                    split="test"
                )
        
        # Create summary report
        self.create_summary_report(results)
        
        return results
    
    def create_summary_report(self, results):
        """Create a summary report of all analyses."""
        print("Creating summary report...")
        
        # Create directory for report
        report_dir = os.path.join(self.output_dir, "summary")
        os.makedirs(report_dir, exist_ok=True)
        
        # Extract key metrics
        summary = {}
        
        # Model comparison metrics
        if "original_vs_detoxified" in results:
            summary["model_comparison"] = {}
            
            # Check the structure of the results
            if isinstance(results["original_vs_detoxified"], dict):
                # If metrics are directly in the dictionary
                if "metrics" in results["original_vs_detoxified"]:
                    metrics_dict = results["original_vs_detoxified"]["metrics"]
                    for model_id in self.models:
                        if model_id in metrics_dict:
                            metrics = metrics_dict[model_id]
                            summary["model_comparison"][model_id] = {
                                "auc_roc": metrics.get("auc_roc", 0),
                                "auc_pr": metrics.get("auc_pr", 0),
                                "accuracy": metrics.get("accuracy", 0),
                                "f1_score": metrics.get("f1_score", 0),
                                "mean_diff": metrics.get("mean_diff", 0)
                            }
                else:
                    # Try to calculate metrics from the raw results
                    original_results = results["original_vs_detoxified"].get("original", {})
                    detoxified_results = results["original_vs_detoxified"].get("detoxified", {})
                    
                    for model_id in self.models:
                        if model_id in original_results and model_id in detoxified_results:
                            # Calculate mean difference
                            orig_scores = original_results[model_id]
                            detox_scores = detoxified_results[model_id]
                            
                            if orig_scores and detox_scores:
                                mean_diff = np.mean(detox_scores) - np.mean(orig_scores)
                                
                                summary["model_comparison"][model_id] = {
                                    "mean_diff": mean_diff,
                                    # Other metrics would require labels, which we don't have here
                                    "auc_roc": 0,
                                    "auc_pr": 0,
                                    "accuracy": 0,
                                    "f1_score": 0
                                }
        
        # Ensemble metrics
        if "ensemble_analysis" in results and "metrics" in results["ensemble_analysis"]:
            ensemble_metrics = results["ensemble_analysis"]["metrics"]
            summary["ensemble_comparison"] = ensemble_metrics
            
            # Find best ensemble method
            if ensemble_metrics:
                # Get all ensemble methods from the metrics
                ensemble_methods = list(ensemble_metrics.keys())
                
                # Find the best method based on AUC-ROC
                best_method = max(ensemble_metrics.keys(), 
                                 key=lambda k: ensemble_metrics[k].get("auc_roc", 0) if k in ensemble_metrics else 0)
                
                summary["best_ensemble_method"] = {
                    "method": best_method,
                    "metrics": ensemble_metrics[best_method]
                }
        
        # Value head analysis
        if "value_head_analysis" in results:
            summary["value_head_analysis"] = results["value_head_analysis"]
        
        # Save summary
        with open(os.path.join(report_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        # Create summary plots
        if "model_comparison" in summary and summary["model_comparison"]:
            # Create bar chart of key metrics
            metrics = ["auc_roc", "auc_pr", "accuracy", "f1_score", "mean_diff"]
            for metric in metrics:
                plt.figure(figsize=(12, 8))
                values = [summary["model_comparison"][model_id].get(metric, 0) for model_id in summary["model_comparison"]]
                plt.bar(list(summary["model_comparison"].keys()), values)
                plt.title(f"{metric.upper()} by Model")
                plt.xlabel("Model")
                plt.ylabel(metric.upper())
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(report_dir, f"{metric}_comparison.png"), dpi=300)
                plt.close()
        
        # Create ensemble comparison plot if available
        if "ensemble_comparison" in summary and summary["ensemble_comparison"]:
            metrics = ["auc_roc", "auc_pr", "accuracy", "f1_score"]
            for metric in metrics:
                plt.figure(figsize=(12, 8))
                values = [summary["ensemble_comparison"][method].get(metric, 0) 
                         for method in summary["ensemble_comparison"]
                         if metric in summary["ensemble_comparison"][method]]
                methods = [method for method in summary["ensemble_comparison"] 
                          if metric in summary["ensemble_comparison"][method]]
                
                if methods:  # Only create plot if we have data
                    plt.bar(methods, values)
                    plt.title(f"{metric.upper()} by Ensemble Method")
                    plt.xlabel("Ensemble Method")
                    plt.ylabel(metric.upper())
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(os.path.join(report_dir, f"ensemble_{metric}_comparison.png"), dpi=300)
                    plt.close()
        
        print(f"Summary report saved to {report_dir}")
        
        return summary

    def load_and_split_paired_datasets(self, original_dataset_path, detoxified_dataset_path, 
                                      train_test_split=0.8, seed=42):
        """
        Load and split paired datasets into train and test sets.
        
        Args:
            original_dataset_path: Path to original (toxic) dataset
            detoxified_dataset_path: Path to detoxified dataset
            train_test_split: Fraction of data to use for training
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with train and test data
        """
        print(f"Loading and splitting paired datasets with split {train_test_split}...")
        
        # Load datasets
        try:
            # Load original dataset
            if original_dataset_path.startswith(("http", "https", "s3://")):
                response = requests.get(original_dataset_path)
                original_data = response.json()
            elif os.path.exists(original_dataset_path):
                with open(original_dataset_path, "r") as f:
                    original_data = json.load(f)
            else:
                # Try to load from HuggingFace
                ds = load_dataset(original_dataset_path)
                if "train" in ds:
                    ds = ds["train"]
                original_data = ds.to_pandas().to_dict(orient="records")
            
            # Load detoxified dataset
            if detoxified_dataset_path.startswith(("http", "https", "s3://")):
                response = requests.get(detoxified_dataset_path)
                detoxified_data = response.json()
            elif os.path.exists(detoxified_dataset_path):
                with open(detoxified_dataset_path, "r") as f:
                    detoxified_data = json.load(f)
            else:
                # Try to load from HuggingFace
                ds = load_dataset(detoxified_dataset_path)
                if "train" in ds:
                    ds = ds["train"]
                detoxified_data = ds.to_pandas().to_dict(orient="records")
            
            # Ensure datasets have the same length
            min_len = min(len(original_data), len(detoxified_data))
            original_data = original_data[:min_len]
            detoxified_data = detoxified_data[:min_len]
            
            # Set random seed for reproducibility
            np.random.seed(seed)
            
            # Create indices for train/test split
            indices = np.random.permutation(min_len)
            train_size = int(train_test_split * min_len)
            
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]
            
            # Split data into train/test sets
            train_data = {
                'original': [original_data[i] for i in train_indices],
                'detoxified': [detoxified_data[i] for i in train_indices]
            }
            
            test_data = {
                'original': [original_data[i] for i in test_indices],
                'detoxified': [detoxified_data[i] for i in test_indices]
            }
            
            print(f"Split datasets into {len(train_data['original'])} train and {len(test_data['original'])} test samples")
            
            return {
                'train': train_data,
                'test': test_data
            }
        
        except Exception as e:
            print(f"Error loading and splitting datasets: {e}")
            return None

    def load_and_split_toxicity_prompts(self, dataset_path="allenai/real-toxicity-prompts", 
                                       toxicity_threshold=0.5, seed=42, test_size=0.2):
        """
        Load and split toxicity prompts dataset similar to RLHF utilities.
        
        Args:
            dataset_path: Path or HuggingFace ID of the dataset
            toxicity_threshold: Threshold for filtering toxic prompts
            seed: Random seed for reproducibility
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with train and test data
        """
        print(f"Loading and splitting toxicity prompts dataset...")
        
        try:
            # Create a mock config for build_dataset
            config = type('Config', (), {
                'model': type('ModelConfig', (), {'name': 'EleutherAI/pythia-70m'}),
                'dataset': type('DatasetConfig', (), {
                    'name': dataset_path,
                    'toxicity_threshold': toxicity_threshold,
                    'input_min_text_length': 32,
                    'input_max_text_length': 256,
                    'test_size': test_size
                }),
                'training': type('TrainingConfig', (), {'seed': seed})
            })
            
            # Use build_dataset from rlhf_utilities
            train_ds, test_ds, tokenizer = build_dataset(config)
            
            print(f"Split toxicity prompts into {len(train_ds)} train and {len(test_ds)} test samples")
            
            return {
                'train': train_ds,
                'test': test_ds,
                'tokenizer': tokenizer
            }
        
        except Exception as e:
            print(f"Error loading and splitting toxicity prompts: {e}")
            return None


class RewardModelEnsemble:
    """Class to create and use an ensemble of reward models."""
    
    def __init__(self, 
                 model_specs: List[Dict[str, Any]],
                 ensemble_method: str = "mean",
                 weights: Optional[Dict[str, float]] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the ensemble.
        
        Args:
            model_specs: List of dictionaries with model specifications
            ensemble_method: Method to use for ensemble (mean, median, max, min, weighted_mean)
            weights: Dictionary mapping model IDs to weights (for weighted_mean)
            device: Device to run models on
        """
        self.model_specs = model_specs
        self.ensemble_method = ensemble_method
        self.device = device
        
        # Initialize weights
        if weights is None:
            # Default to equal weights
            self.weights = {f"pythia-{spec['size']}-seed-{spec['seed']}": 1.0 
                           for spec in model_specs}
        else:
            self.weights = weights
        
        # Normalize weights
        if self.weights:
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        # Load models
        self.models = {}
        self.tokenizers = {}
        self.load_models()
    
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
    parser.add_argument("--train_test_split", type=float, default=0.8,
                        help="Fraction of data to use for training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
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
    
    # Wandb options
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", default="reward-model-ensemble",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", default=None,
                        help="W&B entity name")
    
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
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity
    )
    
    # Run analysis
    results = analyzer.run_full_analysis(
        original_dataset_path=args.original_dataset or f"ajagota71/EleutherAI_pythia-{args.model_sizes[0]}_2000_samples_original",
        detoxified_dataset_path=args.detoxified_dataset or f"ajagota71/ajagota71_pythia-{args.model_sizes[0]}-detox-epoch-100_2000_samples_detoxified",
        toxicity_prompts_dataset_path=args.toxicity_prompts_dataset,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        train_test_split=args.train_test_split,
        seed=args.seed
    )
    
    # Create ensemble if requested
    if not args.analyze_only:
        print("\nCreating reward model ensemble...")
        
        # Get weights from analysis results
        weights = None
        if "test" in results and "ensemble_analysis" in results["test"] and "weights" in results["test"]["ensemble_analysis"]:
            # Prefer weights from test set for better generalization
            weights = results["test"]["ensemble_analysis"]["weights"]
        elif "train" in results and "ensemble_analysis" in results["train"] and "weights" in results["train"]["ensemble_analysis"]:
            weights = results["train"]["ensemble_analysis"]["weights"]
        
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
    
    # Finish wandb run if used
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()