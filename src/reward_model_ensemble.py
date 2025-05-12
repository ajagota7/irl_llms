"""
Script to analyze and ensemble reward models trained with different seeds.
This script investigates how different reward models trained with IRL
might capture different aspects of toxicity detection.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.decomposition import PCA
from scipy import stats
import argparse
import json
from typing import List, Dict, Tuple, Optional, Union
from huggingface_hub import hf_hub_download, list_repo_files

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RewardModelEnsembleAnalyzer:
    """Class to analyze and ensemble reward models trained with different seeds."""

    def __init__(
        self,
        model_size: str,
        seeds: List[int],
        checkpoint: int,
        ground_truth_model: str = "facebook/roberta-hate-speech-dynabench-r4-target",
        original_dataset: str = None,
        detoxified_dataset: str = None,
        batch_size: int = 16,
        max_length: int = 512,
        output_dir: str = "reward_model_analysis",
    ):
        """
        Initialize the analyzer.
        
        Args:
            model_size: Size of the model (70m, 160m, 410m, 1b)
            seeds: List of seeds to analyze
            checkpoint: Checkpoint number to use
            ground_truth_model: HF model ID for ground truth toxicity classifier
            original_dataset: HF dataset ID for original (toxic) outputs
            detoxified_dataset: HF dataset ID for detoxified outputs
            batch_size: Batch size for inference
            max_length: Max sequence length for tokenization
            output_dir: Directory to save analysis results
        """
        self.model_size = model_size
        self.seeds = seeds
        self.checkpoint = checkpoint
        self.ground_truth_model_name = ground_truth_model
        
        if original_dataset is None:
            self.original_dataset = f"ajagota71/EleutherAI_pythia-{model_size}_2000_samples_original"
        else:
            self.original_dataset = original_dataset
            
        if detoxified_dataset is None:
            self.detoxified_dataset = f"ajagota71/ajagota71_pythia-{model_size}-detox-epoch-100_2000_samples_detoxified"
        else:
            self.detoxified_dataset = detoxified_dataset
            
        self.batch_size = batch_size
        self.max_length = max_length
        self.output_dir = os.path.join(output_dir, f"pythia-{model_size}")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load models and data
        self._load_reward_models()
        self._load_ground_truth_model()
        self._load_data()
        
    def _load_reward_models(self):
        """Load all reward models for the specified seeds."""
        print(f"Loading reward models for pythia-{self.model_size}...")
        
        self.reward_models = {}
        self.reward_tokenizers = {}
        
        for seed in tqdm(self.seeds, desc="Loading models"):
            model_id = f"ajagota71/toxicity-reward-model-max-margin-seed-{seed}-pythia-{self.model_size}-checkpoint-{self.checkpoint}"
            
            try:
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = 'left'
                
                # Load model
                model = AutoModelForSequenceClassification.from_pretrained(model_id)
                model.to(device)
                model.eval()
                
                self.reward_models[seed] = model
                self.reward_tokenizers[seed] = tokenizer
                
                print(f"Successfully loaded model for seed {seed}")
            except Exception as e:
                print(f"Error loading model for seed {seed}: {e}")
                continue
                
        print(f"Loaded {len(self.reward_models)} reward models")
        
    def _load_ground_truth_model(self):
        """Load the ground truth toxicity model."""
        print(f"Loading ground truth model: {self.ground_truth_model_name}")
        
        try:
            self.ground_truth_tokenizer = AutoTokenizer.from_pretrained(self.ground_truth_model_name)
            self.ground_truth_model = AutoModelForSequenceClassification.from_pretrained(
                self.ground_truth_model_name
            ).to(device)
            self.ground_truth_model.eval()
            print("Successfully loaded ground truth model")
        except Exception as e:
            print(f"Error loading ground truth model: {e}")
            self.ground_truth_model = None
            self.ground_truth_tokenizer = None
            
    def _load_data(self):
        """Load the original and detoxified datasets."""
        print("Loading datasets...")
        
        try:
            # Define a helper function to load datasets
            def safe_load_dataset(dataset_path):
                print(f"Loading dataset: {dataset_path}")
                
                # Try multiple methods to load the dataset
                try:
                    # Method 1: Standard loading
                    try:
                        ds = load_dataset(dataset_path, download_mode="force_redownload")
                        if isinstance(ds, dict) and 'train' in ds:
                            return ds['train']
                        return ds
                    except Exception as e1:
                        print(f"Standard loading failed: {e1}")
                
                    # Method 2: With auth token
                    try:
                        ds = load_dataset(dataset_path, use_auth_token=True)
                        if isinstance(ds, dict) and 'train' in ds:
                            return ds['train']
                        return ds
                    except Exception as e2:
                        print(f"Loading with auth token failed: {e2}")
                
                    # Method 3: Streaming mode
                    try:
                        ds = load_dataset(dataset_path, streaming=True)
                        return list(ds.take(10000))  # Take a large number to ensure we get all data
                    except Exception as e3:
                        print(f"Streaming mode failed: {e3}")
                
                    # Method 4: Direct file download
                    files = list_repo_files(dataset_path, repo_type="dataset")
                    print(f"Files in repository: {files}")
                    
                    # Look for data files
                    data_files = [f for f in files if f.endswith(('.json', '.csv', '.parquet', '.jsonl'))]
                    
                    if not data_files:
                        raise ValueError(f"No data files found in repository: {dataset_path}")
                        
                    # Download the first data file
                    file_path = hf_hub_download(
                        repo_id=dataset_path,
                        filename=data_files[0],
                        repo_type="dataset"
                    )
                    
                    # Load based on file extension
                    if file_path.endswith('.json') or file_path.endswith('.jsonl'):
                        with open(file_path, 'r') as f:
                            return json.load(f)
                    elif file_path.endswith('.csv'):
                        return pd.read_csv(file_path).to_dict('records')
                    elif file_path.endswith('.parquet'):
                        return pd.read_parquet(file_path).to_dict('records')
                    else:
                        raise ValueError(f"Unsupported file format: {file_path}")
                        
                except Exception as e:
                    print(f"All loading methods failed: {e}")
                    raise ValueError(f"Could not load dataset: {dataset_path}")
            
            # Load original dataset
            original_data = safe_load_dataset(self.original_dataset)
            
            # Convert to list if it's a Dataset object
            if hasattr(original_data, 'to_pandas'):
                self.original_data = original_data.to_pandas().to_dict('records')
            else:
                self.original_data = original_data
            
            # Load detoxified dataset
            detoxified_data = safe_load_dataset(self.detoxified_dataset)
            
            # Convert to list if it's a Dataset object
            if hasattr(detoxified_data, 'to_pandas'):
                self.detoxified_data = detoxified_data.to_pandas().to_dict('records')
            else:
                self.detoxified_data = detoxified_data
            
            # Verify data lengths match
            if len(self.original_data) != len(self.detoxified_data):
                print("Warning: Dataset lengths don't match!")
                # Use the smaller length
                min_len = min(len(self.original_data), len(self.detoxified_data))
                self.original_data = self.original_data[:min_len]
                self.detoxified_data = self.detoxified_data[:min_len]
            
            print(f"Loaded {len(self.original_data)} paired samples")
            
            # Create a combined dataset with labels
            self.all_texts = []
            self.all_labels = []  # 1 for toxic (original), 0 for non-toxic (detoxified)
            
            # Add original (toxic) examples
            for item in self.original_data:
                self.all_texts.append(item['output'])
                self.all_labels.append(1)  # Toxic
            
            # Add detoxified examples
            for item in self.detoxified_data:
                self.all_texts.append(item['output'])
                self.all_labels.append(0)  # Non-toxic
                
        except Exception as e:
            print(f"Error loading datasets: {e}")
            raise
        
    def get_model_predictions(self, texts: List[str], model, tokenizer) -> np.ndarray:
        """
        Get predictions from a model for a list of texts.
        
        Args:
            texts: List of text strings
            model: The model to use for predictions
            tokenizer: The tokenizer for the model
            
        Returns:
            Array of prediction scores
        """
        all_predictions = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(device)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                
            # For reward models, we expect a single score
            if hasattr(outputs, "rewards"):
                scores = outputs.rewards.squeeze().cpu().numpy()
            # For classification models like the ground truth model
            elif hasattr(outputs, "logits"):
                # Use the first logit (non-toxic) as the score
                scores = outputs.logits[:, 0].cpu().numpy()
            else:
                raise ValueError(f"Unexpected model output format: {type(outputs)}")
                
            # Handle single item case
            if not isinstance(scores, np.ndarray):
                scores = np.array([scores])
                
            all_predictions.extend(scores)
            
        return np.array(all_predictions)
        
    def analyze_model_correlations(self):
        """
        Analyze correlations between different reward models and the ground truth.
        """
        print("Analyzing model correlations...")
        
        # Get a subset of texts for correlation analysis (for efficiency)
        sample_size = min(1000, len(self.all_texts))
        sample_indices = np.random.choice(len(self.all_texts), sample_size, replace=False)
        sample_texts = [self.all_texts[i] for i in sample_indices]
        sample_labels = [self.all_labels[i] for i in sample_indices]
        
        # Get predictions from all models
        predictions = {}
        
        # Get ground truth predictions if available
        if self.ground_truth_model is not None:
            ground_truth_preds = self.get_model_predictions(
                sample_texts, 
                self.ground_truth_model, 
                self.ground_truth_tokenizer
            )
            predictions["ground_truth"] = ground_truth_preds
            
        # Get predictions from each reward model
        for seed, model in self.reward_models.items():
            model_preds = self.get_model_predictions(
                sample_texts, 
                model, 
                self.reward_tokenizers[seed]
            )
            predictions[f"seed_{seed}"] = model_preds
            
        # Create a DataFrame for correlation analysis
        df = pd.DataFrame(predictions)
        
        # Add true labels
        df["true_label"] = sample_labels
        
        # Calculate correlation matrix
        correlation_matrix = df.corr(method='pearson')
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'Correlation Between Reward Models (Pythia-{self.model_size})')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_correlations.png'))
        plt.close()
        
        # Save correlation matrix
        correlation_matrix.to_csv(os.path.join(self.output_dir, 'model_correlations.csv'))
        
        # Calculate average correlation between models
        model_columns = [col for col in df.columns if col.startswith('seed_')]
        model_corr = correlation_matrix.loc[model_columns, model_columns]
        avg_model_corr = (model_corr.sum().sum() - len(model_columns)) / (len(model_columns) * (len(model_columns) - 1))
        
        print(f"Average correlation between models: {avg_model_corr:.4f}")
        
        # Calculate correlation with ground truth if available
        if "ground_truth" in df.columns:
            gt_corr = correlation_matrix.loc["ground_truth", model_columns].mean()
            print(f"Average correlation with ground truth: {gt_corr:.4f}")
            
        # Calculate correlation with true labels
        label_corr = correlation_matrix.loc["true_label", model_columns].mean()
        print(f"Average correlation with true labels: {label_corr:.4f}")
        
        return correlation_matrix
        
    def analyze_feature_representations(self):
        """
        Analyze the feature representations learned by different models.
        Uses PCA to visualize the embeddings from different models.
        """
        print("Analyzing feature representations...")
        
        # Sample a smaller subset for feature analysis
        sample_size = min(500, len(self.all_texts))
        sample_indices = np.random.choice(len(self.all_texts), sample_size, replace=False)
        sample_texts = [self.all_texts[i] for i in sample_indices]
        sample_labels = [self.all_labels[i] for i in sample_indices]
        
        # Function to extract embeddings from a model
        def get_embeddings(model, tokenizer, texts):
            embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i+self.batch_size]
                
                # Tokenize
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                ).to(device)
                
                # Get embeddings from the model
                with torch.no_grad():
                    # Forward pass but extract the last hidden state
                    outputs = model(**inputs, output_hidden_states=True)
                    
                    # Get the last hidden state (CLS token embedding)
                    if hasattr(outputs, "hidden_states"):
                        # Get the last layer's [CLS] token embedding
                        last_hidden_state = outputs.hidden_states[-1]
                        cls_embeddings = last_hidden_state[:, 0, :].cpu().numpy()
                        embeddings.extend(cls_embeddings)
                    else:
                        print(f"Model doesn't output hidden states: {type(outputs)}")
                        return None
                        
            return np.array(embeddings)
        
        # Get embeddings from each model
        all_embeddings = {}
        for seed, model in self.reward_models.items():
            print(f"Extracting embeddings for seed {seed}...")
            embeddings = get_embeddings(model, self.reward_tokenizers[seed], sample_texts)
            if embeddings is not None:
                all_embeddings[f"seed_{seed}"] = embeddings
        
        # Apply PCA to each model's embeddings
        pca_results = {}
        for model_name, embeddings in all_embeddings.items():
            print(f"Applying PCA to {model_name} embeddings...")
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(embeddings)
            pca_results[model_name] = pca_result
            print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        
        # Plot PCA results
        plt.figure(figsize=(15, 10))
        
        for i, (model_name, pca_result) in enumerate(pca_results.items()):
            plt.subplot(2, (len(pca_results) + 1) // 2, i + 1)
            
            # Plot points colored by true label
            scatter = plt.scatter(
                pca_result[:, 0], 
                pca_result[:, 1], 
                c=sample_labels, 
                cmap='coolwarm', 
                alpha=0.6
            )
            
            plt.colorbar(scatter, label='True Label (1=Toxic)')
            plt.title(f'PCA of {model_name} Embeddings')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_representations_pca.png'))
        plt.close()
        
        return pca_results
        
    def evaluate_individual_models(self):
        """
        Evaluate each individual model's performance on the dataset.
        """
        print("Evaluating individual models...")
        
        results = {}
        
        # Evaluate each reward model
        for seed, model in self.reward_models.items():
            print(f"Evaluating model with seed {seed}...")
            
            # Get predictions
            predictions = self.get_model_predictions(
                self.all_texts, 
                model, 
                self.reward_tokenizers[seed]
            )
            
            # Normalize predictions (higher = less toxic)
            # Convert to binary predictions using mean threshold
            threshold = np.mean(predictions)
            binary_preds = (predictions > threshold).astype(int)
            binary_preds = 1 - binary_preds  # Invert to match ground truth (1=toxic)
            
            # Calculate metrics
            accuracy = accuracy_score(self.all_labels, binary_preds)
            f1 = f1_score(self.all_labels, binary_preds)
            auc_roc = roc_auc_score(self.all_labels, [-x for x in predictions])  # Invert for ROC
            
            # Store results
            results[seed] = {
                "accuracy": accuracy,
                "f1": f1,
                "auc_roc": auc_roc,
                "predictions": predictions.tolist(),
                "binary_predictions": binary_preds.tolist()
            }
            
            print(f"Seed {seed} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC-ROC: {auc_roc:.4f}")
            
        # Save results
        with open(os.path.join(self.output_dir, 'individual_model_results.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for seed, result in results.items():
                serializable_results[str(seed)] = {
                    k: v for k, v in result.items() 
                    if k not in ["predictions", "binary_predictions"]
                }
            json.dump(serializable_results, f, indent=2)
            
        # Plot performance comparison
        seeds = list(results.keys())
        accuracies = [results[seed]["accuracy"] for seed in seeds]
        f1_scores = [results[seed]["f1"] for seed in seeds]
        auc_scores = [results[seed]["auc_roc"] for seed in seeds]
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(seeds))
        width = 0.25
        
        plt.bar(x - width, accuracies, width, label='Accuracy')
        plt.bar(x, f1_scores, width, label='F1 Score')
        plt.bar(x + width, auc_scores, width, label='AUC-ROC')
        
        plt.xlabel('Seed')
        plt.ylabel('Score')
        plt.title(f'Performance Metrics by Seed (Pythia-{self.model_size})')
        plt.xticks(x, seeds)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'individual_model_performance.png'))
        plt.close()
        
        return results
        
    def evaluate_ensemble_methods(self):
        """
        Evaluate different ensemble methods for combining model predictions.
        """
        print("Evaluating ensemble methods...")
        
        # Get predictions from all models
        all_predictions = {}
        for seed, model in self.reward_models.items():
            predictions = self.get_model_predictions(
                self.all_texts, 
                model, 
                self.reward_tokenizers[seed]
            )
            all_predictions[seed] = predictions
            
        # Convert to numpy array for easier manipulation
        # Shape: (num_models, num_samples)
        predictions_array = np.array([all_predictions[seed] for seed in self.seeds])
        
        # Define ensemble methods
        ensemble_methods = {
            "mean": np.mean(predictions_array, axis=0),
            "median": np.median(predictions_array, axis=0),
            "max": np.max(predictions_array, axis=0),
            "min": np.min(predictions_array, axis=0)
        }
        
        # Evaluate each ensemble method
        ensemble_results = {}
        
        for method_name, ensemble_preds in ensemble_methods.items():
            # Normalize predictions (higher = less toxic)
            # Convert to binary predictions using mean threshold
            threshold = np.mean(ensemble_preds)
            binary_preds = (ensemble_preds > threshold).astype(int)
            binary_preds = 1 - binary_preds  # Invert to match ground truth (1=toxic)
            
            # Calculate metrics
            accuracy = accuracy_score(self.all_labels, binary_preds)
            f1 = f1_score(self.all_labels, binary_preds)
            auc_roc = roc_auc_score(self.all_labels, [-x for x in ensemble_preds])  # Invert for ROC
            
            # Store results
            ensemble_results[method_name] = {
                "accuracy": accuracy,
                "f1": f1,
                "auc_roc": auc_roc
            }
            
            print(f"Ensemble ({method_name}) - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC-ROC: {auc_roc:.4f}")
            
        # Save results
        with open(os.path.join(self.output_dir, 'ensemble_results.json'), 'w') as f:
            json.dump(ensemble_results, f, indent=2)
            
        # Plot performance comparison
        methods = list(ensemble_results.keys())
        accuracies = [ensemble_results[method]["accuracy"] for method in methods]
        f1_scores = [ensemble_results[method]["f1"] for method in methods]
        auc_scores = [ensemble_results[method]["auc_roc"] for method in methods]
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(methods))
        width = 0.25
        
        plt.bar(x - width, accuracies, width, label='Accuracy')
        plt.bar(x, f1_scores, width, label='F1 Score')
        plt.bar(x + width, auc_scores, width, label='AUC-ROC')
        
        plt.xlabel('Ensemble Method')
        plt.ylabel('Score')
        plt.title(f'Ensemble Performance Metrics (Pythia-{self.model_size})')
        plt.xticks(x, methods)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ensemble_performance.png'))
        plt.close()
        
        # Compare best ensemble with best individual model
        individual_results = self.evaluate_individual_models()
        best_individual_seed = max(individual_results.keys(), key=lambda s: individual_results[s]["auc_roc"])
        best_individual = individual_results[best_individual_seed]
        
        best_ensemble_method = max(ensemble_results.keys(), key=lambda m: ensemble_results[m]["auc_roc"])
        best_ensemble = ensemble_results[best_ensemble_method]
        
        comparison = {
            "best_individual": {
                "seed": best_individual_seed,
                **best_individual
            },
            "best_ensemble": {
                "method": best_ensemble_method,
                **best_ensemble
            }
        }
        
        # Save comparison
        with open(os.path.join(self.output_dir, 'best_model_comparison.json'), 'w') as f:
            # Remove predictions from the output
            if "predictions" in comparison["best_individual"]:
                del comparison["best_individual"]["predictions"]
            if "binary_predictions" in comparison["best_individual"]:
                del comparison["best_individual"]["binary_predictions"]
            json.dump(comparison, f, indent=2)
            
        print("\nBest Model Comparison:")
        print(f"Best Individual Model (Seed {best_individual_seed}):")
        print(f"  Accuracy: {best_individual['accuracy']:.4f}")
        print(f"  F1 Score: {best_individual['f1']:.4f}")
        print(f"  AUC-ROC: {best_individual['auc_roc']:.4f}")
        
        print(f"\nBest Ensemble Method ({best_ensemble_method}):")
        print(f"  Accuracy: {best_ensemble['accuracy']:.4f}")
        print(f"  F1 Score: {best_ensemble['f1']:.4f}")
        print(f"  AUC-ROC: {best_ensemble['auc_roc']:.4f}")
        
        return ensemble_results, comparison
        
    def analyze_error_patterns(self):
        """
        Analyze error patterns across different models to see if they make different types of errors.
        """
        print("Analyzing error patterns...")
        
        # Get predictions from all models
        all_predictions = {}
        all_binary_preds = {}
        
        for seed, model in self.reward_models.items():
            predictions = self.get_model_predictions(
                self.all_texts, 
                model, 
                self.reward_tokenizers[seed]
            )
            all_predictions[seed] = predictions
            
            # Convert to binary predictions
            threshold = np.mean(predictions)
            binary_preds = (predictions > threshold).astype(int)
            binary_preds = 1 - binary_preds  # Invert to match ground truth (1=toxic)
            all_binary_preds[seed] = binary_preds
            
        # Find examples where models disagree
        num_models = len(self.seeds)
        disagreement_counts = np.zeros(len(self.all_texts))
        
        for i in range(len(self.all_texts)):
            # Count unique predictions for this example
            unique_preds = set(all_binary_preds[seed][i] for seed in self.seeds)
            if len(unique_preds) > 1:  # Models disagree
                disagreement_counts[i] = 1
                
        # Calculate disagreement rate
        disagreement_rate = disagreement_counts.mean()
        print(f"Model disagreement rate: {disagreement_rate:.4f}")
        
        # Find examples where all models are wrong
        all_wrong_indices = []
        for i in range(len(self.all_texts)):
            if all(all_binary_preds[seed][i] != self.all_labels[i] for seed in self.seeds):
                all_wrong_indices.append(i)
                
        print(f"Number of examples where all models are wrong: {len(all_wrong_indices)}")
        
        # Find examples where only some models are right
        some_right_indices = []
        for i in range(len(self.all_texts)):
            correct_preds = sum(all_binary_preds[seed][i] == self.all_labels[i] for seed in self.seeds)
            if 0 < correct_preds < num_models:
                some_right_indices.append(i)
                
        print(f"Number of examples where only some models are right: {len(some_right_indices)}")
        
        # Calculate confusion matrices for each model
        confusion_matrices = {}
        for seed in self.seeds:
            cm = confusion_matrix(self.all_labels, all_binary_preds[seed])
            confusion_matrices[seed] = cm
            
        # Plot confusion matrices
        fig, axes = plt.subplots(1, len(self.seeds), figsize=(5*len(self.seeds), 5))
        if len(self.seeds) == 1:
            axes = [axes]
            
        for i, seed in enumerate(self.seeds):
            cm = confusion_matrices[seed]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'Seed {seed}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('True')
            axes[i].set_xticklabels(['Non-toxic', 'Toxic'])
            axes[i].set_yticklabels(['Non-toxic', 'Toxic'])
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrices.png'))
        plt.close()
        
        # Save some example texts where models disagree
        if len(some_right_indices) > 0:
            num_examples = min(10, len(some_right_indices))
            example_indices = np.random.choice(some_right_indices, num_examples, replace=False)
            
            examples = []
            for idx in example_indices:
                example = {
                    "text": self.all_texts[idx],
                    "true_label": self.all_labels[idx],
                    "predictions": {seed: int(all_binary_preds[seed][idx]) for seed in self.seeds},
                    "scores": {seed: float(all_predictions[seed][idx]) for seed in self.seeds}
                }
                examples.append(example)
                
            with open(os.path.join(self.output_dir, 'disagreement_examples.json'), 'w') as f:
                json.dump(examples, f, indent=2)
                
        return {
            "disagreement_rate": disagreement_rate,
            "all_wrong_count": len(all_wrong_indices),
            "some_right_count": len(some_right_indices),
            "confusion_matrices": {seed: cm.tolist() for seed, cm in confusion_matrices.items()}
        }
        
    def run_full_analysis(self):
        """Run all analysis methods and compile results."""
        print(f"Running full analysis for pythia-{self.model_size}...")
        
        results = {}
        
        # Run all analyses
        results["correlations"] = self.analyze_model_correlations()
        self.analyze_feature_representations()
        results["individual_performance"] = self.evaluate_individual_models()
        results["ensemble_performance"], results["best_comparison"] = self.evaluate_ensemble_methods()
        results["error_patterns"] = self.analyze_error_patterns()
        
        # Save summary results
        summary = {
            "model_size": self.model_size,
            "seeds": self.seeds,
            "num_models": len(self.reward_models),
            "dataset_size": len(self.all_texts) // 2,  # Divide by 2 because we have original + detoxified
        }
        
        if isinstance(results["correlations"], pd.DataFrame):
            # Extract key correlation metrics
            if len(model_cols) > 1:
                model_corr = results["correlations"].loc[model_cols, model_cols]
                avg_model_corr = (model_corr.sum().sum() - len(model_cols)) / (len(model_cols) * (len(model_cols) - 1))
                summary["average_model_correlation"] = avg_model_corr
                
            if "ground_truth" in results["correlations"].columns:
                gt_corr = results["correlations"].loc["ground_truth", model_cols].mean()
                summary["average_ground_truth_correlation"] = gt_corr
                
            if "true_label" in results["correlations"].columns:
                label_corr = results["correlations"].loc["true_label", model_cols].mean()
                summary["average_label_correlation"] = label_corr
        
        # Add ensemble improvement metrics
        if "best_comparison" in results:
            best_individual = results["best_comparison"]["best_individual"]
            best_ensemble = results["best_comparison"]["best_ensemble"]
            
            # Calculate improvement percentages
            accuracy_improvement = (best_ensemble["accuracy"] - best_individual["accuracy"]) / best_individual["accuracy"] * 100
            f1_improvement = (best_ensemble["f1"] - best_individual["f1"]) / best_individual["f1"] * 100
            auc_improvement = (best_ensemble["auc_roc"] - best_individual["auc_roc"]) / best_individual["auc_roc"] * 100
            
            summary["ensemble_improvements"] = {
                "accuracy_improvement_percent": accuracy_improvement,
                "f1_improvement_percent": f1_improvement,
                "auc_improvement_percent": auc_improvement,
                "best_individual_seed": best_individual["seed"],
                "best_ensemble_method": best_ensemble["method"]
            }
            
        # Add error pattern metrics
        if "error_patterns" in results:
            summary["disagreement_rate"] = results["error_patterns"]["disagreement_rate"]
            summary["all_wrong_count"] = results["error_patterns"]["all_wrong_count"]
            summary["some_right_count"] = results["error_patterns"]["some_right_count"]
            
        # Save summary
        with open(os.path.join(self.output_dir, 'analysis_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
            
        print("\nAnalysis Summary:")
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, float):
                        print(f"  {subkey}: {subvalue:.4f}")
                    else:
                        print(f"  {subkey}: {subvalue}")
            elif isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
                
        return summary


def analyze_across_model_sizes(
    model_sizes: List[str],
    seeds: List[int],
    checkpoints: Dict[str, int],
    output_dir: str = "reward_model_analysis"
):
    """
    Run analysis across different model sizes and compare results.
    
    Args:
        model_sizes: List of model sizes to analyze
        seeds: List of seeds to use for each model size
        checkpoints: Dictionary mapping model sizes to checkpoint numbers
        output_dir: Directory to save analysis results
    """
    print(f"Analyzing across model sizes: {model_sizes}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run analysis for each model size
    summaries = {}
    
    for model_size in model_sizes:
        print(f"\n{'='*50}")
        print(f"Analyzing model size: {model_size}")
        print(f"{'='*50}\n")
        
        checkpoint = checkpoints.get(model_size, 30)  # Default to 30 if not specified
        
        analyzer = RewardModelEnsembleAnalyzer(
            model_size=model_size,
            seeds=seeds,
            checkpoint=checkpoint,
            output_dir=output_dir
        )
        
        summary = analyzer.run_full_analysis()
        summaries[model_size] = summary
    
    # Compare results across model sizes
    print("\nComparing results across model sizes...")
    
    # Extract key metrics for comparison
    comparison = {
        "model_sizes": model_sizes,
        "seeds": seeds,
        "metrics": {}
    }
    
    # Define metrics to compare
    metrics_to_compare = [
        "average_model_correlation",
        "average_ground_truth_correlation",
        "average_label_correlation",
        "disagreement_rate"
    ]
    
    # Add ensemble improvement metrics
    ensemble_metrics = [
        "accuracy_improvement_percent",
        "f1_improvement_percent",
        "auc_improvement_percent"
    ]
    
    # Extract metrics
    for metric in metrics_to_compare:
        comparison["metrics"][metric] = {
            size: summaries[size].get(metric, None) for size in model_sizes
        }
        
    # Extract ensemble improvement metrics
    for metric in ensemble_metrics:
        comparison["metrics"][metric] = {
            size: summaries[size].get("ensemble_improvements", {}).get(metric, None) 
            for size in model_sizes
        }
    
    # Save comparison
    with open(os.path.join(output_dir, 'cross_model_comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Plot key metrics across model sizes
    for metric in metrics_to_compare + ensemble_metrics:
        values = [comparison["metrics"][metric].get(size, None) for size in model_sizes]
        
        # Skip if any values are None
        if None in values:
            continue
            
        plt.figure(figsize=(10, 6))
        plt.bar(model_sizes, values)
        plt.xlabel('Model Size')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} Across Model Sizes')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{metric}.png'))
        plt.close()
    
    print("\nCross-model comparison complete!")
    return comparison


def weighted_ensemble_predictions(
    model_size: str,
    seeds: List[int],
    checkpoint: int,
    weights: Optional[Dict[int, float]] = None,
    texts: List[str] = None,
    dataset_path: Optional[str] = None
):
    """
    Generate predictions using a weighted ensemble of reward models.
    
    Args:
        model_size: Size of the model (70m, 160m, 410m, 1b)
        seeds: List of seeds to use in the ensemble
        checkpoint: Checkpoint number to use
        weights: Optional dictionary mapping seeds to weights (defaults to equal weights)
        texts: List of texts to generate predictions for
        dataset_path: Optional path to a dataset to use instead of texts
        
    Returns:
        Array of ensemble predictions
    """
    # Initialize analyzer to load models
    analyzer = RewardModelEnsembleAnalyzer(
        model_size=model_size,
        seeds=seeds,
        checkpoint=checkpoint,
        output_dir="temp"  # Temporary directory
    )
    
    # Use equal weights if not provided
    if weights is None:
        weights = {seed: 1.0 / len(seeds) for seed in seeds}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {seed: weight / total_weight for seed, weight in weights.items()}
    
    # Load texts from dataset if provided
    if texts is None and dataset_path is not None:
        try:
            ds = load_dataset(dataset_path)
            if isinstance(ds, dict) and 'train' in ds:
                ds = ds['train']
            
            # Extract texts from dataset
            if 'text' in ds.column_names:
                texts = ds['text']
            elif 'output' in ds.column_names:
                texts = ds['output']
            else:
                raise ValueError(f"Could not find text column in dataset: {dataset_path}")
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    if texts is None:
        raise ValueError("Either texts or dataset_path must be provided")
    
    # Get predictions from each model
    all_predictions = {}
    for seed, model in analyzer.reward_models.items():
        if seed in weights:
            predictions = analyzer.get_model_predictions(
                texts, 
                model, 
                analyzer.reward_tokenizers[seed]
            )
            all_predictions[seed] = predictions
    
    # Calculate weighted ensemble predictions
    ensemble_preds = np.zeros(len(texts))
    for seed, preds in all_predictions.items():
        ensemble_preds += weights[seed] * preds
    
    return ensemble_preds


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description="Analyze reward model ensembles")
    parser.add_argument("--model_sizes", nargs="+", default=["70m", "160m", "410m", "1b"],
                        help="Model sizes to analyze")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 200, 300, 400],
                        help="Seeds to analyze")
    parser.add_argument("--output_dir", default="reward_model_analysis",
                        help="Directory to save analysis results")
    parser.add_argument("--cross_model", action="store_true",
                        help="Run analysis across model sizes")
    
    args = parser.parse_args()
    
    # Define checkpoints for each model size
    checkpoints = {
        "70m": 30,
        "160m": 50,
        "410m": 70,
        "1b": 70
    }
    
    if args.cross_model:
        # Run analysis across model sizes
        analyze_across_model_sizes(
            model_sizes=args.model_sizes,
            seeds=args.seeds,
            checkpoints=checkpoints,
            output_dir=args.output_dir
        )
    else:
        # Run analysis for each model size separately
        for model_size in args.model_sizes:
            print(f"\n{'='*50}")
            print(f"Analyzing model size: {model_size}")
            print(f"{'='*50}\n")
            
            checkpoint = checkpoints.get(model_size, 30)  # Default to 30 if not specified
            
            analyzer = RewardModelEnsembleAnalyzer(
                model_size=model_size,
                seeds=args.seeds,
                checkpoint=checkpoint,
                output_dir=args.output_dir
            )
            
            analyzer.run_full_analysis()


if __name__ == "__main__":
    main() 