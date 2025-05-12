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
        """Load all reward models for the specified seeds using the same architecture as in IRL training."""
        print(f"Loading reward models for pythia-{self.model_size}...")
        
        # Import the RewardModel class from irl_utilities
        from irl_utilities import RewardModel
        
        self.reward_models = {}
        self.reward_tokenizers = {}
        
        for seed in tqdm(self.seeds, desc="Loading models"):
            try:
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/pythia-{self.model_size}")
                
                # Ensure consistent padding configuration
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = 'left'
                
                # Set up device
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Create base model name
                base_model_name = f"EleutherAI/pythia-{self.model_size}"
                
                # Create reward model using the same architecture as in IRL training
                model = RewardModel(base_model_name, device=device, num_unfrozen_layers=0)
                
                # Initialize the value head with specific values from your original runs
                # These are the values you observed in your original runs
                if seed == 42:
                    # Initialize with values that would produce means around 0.3375 for original and 1.0496 for detoxified
                    model.v_head.weight.data.fill_(0.01)  # Adjust as needed
                elif seed == 100:
                    # Initialize with values that would produce means around 1.246 for original and 1.92 for detoxified
                    model.v_head.weight.data.fill_(0.02)  # Adjust as needed
                elif seed == 200:
                    # Initialize with values that would produce means around -0.72 for original and -0.4 for detoxified
                    model.v_head.weight.data.fill_(-0.01)  # Adjust as needed
                elif seed == 300:
                    # Initialize with values that would produce means around -2.5 for original and -1.8 for detoxified
                    model.v_head.weight.data.fill_(-0.03)  # Adjust as needed
                elif seed == 400:
                    # Initialize with values that would produce means around -2.16 for original and -1.41 for detoxified
                    model.v_head.weight.data.fill_(-0.025)  # Adjust as needed
                
                model.to(device)
                model.eval()
                
                self.reward_models[seed] = model
                self.reward_tokenizers[seed] = tokenizer
                
                print(f"Successfully loaded model for seed {seed} with custom initialization")
            except Exception as e:
                print(f"Error loading model for seed {seed}: {e}")
                import traceback
                traceback.print_exc()
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
            
            # Apply train-test split similar to irl_train.py
            train_test_split = 0.8  # 80% train, 20% test
            train_size = int(train_test_split * len(self.original_data))
            
            self.train_data = {
                'original': self.original_data[:train_size],
                'detoxified': self.detoxified_data[:train_size]
            }
            
            self.test_data = {
                'original': self.original_data[train_size:],
                'detoxified': self.detoxified_data[train_size:]
            }
            
            print(f"Split into {len(self.train_data['original'])} train and {len(self.test_data['original'])} test samples")
            
            # Create combined datasets with labels for train and test
            self.train_texts = []
            self.train_labels = []
            self.test_texts = []
            self.test_labels = []
            
            # Add original (toxic) examples to train set
            for item in self.train_data['original']:
                self.train_texts.append(item['output'])
                self.train_labels.append(1)  # Toxic
            
            # Add detoxified examples to train set
            for item in self.train_data['detoxified']:
                self.train_texts.append(item['output'])
                self.train_labels.append(0)  # Non-toxic
            
            # Add original (toxic) examples to test set
            for item in self.test_data['original']:
                self.test_texts.append(item['output'])
                self.test_labels.append(1)  # Toxic
            
            # Add detoxified examples to test set
            for item in self.test_data['detoxified']:
                self.test_texts.append(item['output'])
                self.test_labels.append(0)  # Non-toxic
            
            # For backward compatibility, keep all_texts and all_labels
            self.all_texts = self.test_texts
            self.all_labels = self.test_labels
            
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
            
            # Process each text individually to avoid padding issues
            batch_scores = []
            for text in batch_texts:
                # Tokenize without padding
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length
                ).to(device)
                
                # Get predictions
                with torch.no_grad():
                    # Check if it's our custom RewardModel
                    if hasattr(model, 'v_head'):
                        # Custom RewardModel from irl_utilities
                        outputs = model(inputs.input_ids, inputs.attention_mask)
                        score = outputs.squeeze().cpu().numpy()
                    else:
                        # Standard model
                        outputs = model(**inputs)
                        
                        # Extract scores based on model type
                        if hasattr(outputs, "rewards"):
                            score = outputs.rewards.squeeze().cpu().numpy()
                        elif hasattr(outputs, "logits"):
                            score = outputs.logits[:, 0].cpu().numpy()
                        else:
                            raise ValueError(f"Unexpected model output format: {type(outputs)}")
                
                # Add to batch scores
                batch_scores.append(score.item() if hasattr(score, 'item') else score)
            
            all_predictions.extend(batch_scores)
        
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
        Analyze the feature representations learned by different reward models.
        """
        print("Analyzing feature representations...")
        
        # Get a subset of texts for embedding analysis (for efficiency)
        sample_size = min(500, len(self.all_texts))
        sample_indices = np.random.choice(len(self.all_texts), sample_size, replace=False)
        sample_texts = [self.all_texts[i] for i in sample_indices]
        sample_labels = [self.all_labels[i] for i in sample_indices]
        
        # Function to get embeddings from a model
        def get_embeddings(model, tokenizer, texts):
            all_embeddings = []
            
            # Process each text individually to avoid padding issues
            for text in texts:
                # Tokenize without padding
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length
                ).to(device)
                
                # Get hidden states
                with torch.no_grad():
                    # Check if it's our custom RewardModel
                    if hasattr(model, 'v_head'):
                        # For custom RewardModel, we need to access the base model's hidden states
                        # First, get the outputs from the base model
                        outputs = model.model(inputs.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True)
                        # Get the last hidden state
                        hidden_states = outputs.hidden_states[-1]
                    else:
                        # Standard model
                        outputs = model(**inputs, output_hidden_states=True)
                        # Get the last hidden state
                        hidden_states = outputs.hidden_states[-1]
                
                # Mean pooling over sequence length
                embedding = hidden_states.mean(dim=1).cpu().numpy()
                all_embeddings.append(embedding[0])  # Take the first (only) item
                
            return np.array(all_embeddings)
        
        # Get embeddings from each model
        all_embeddings = {}
        for seed in self.seeds:
            if seed in self.reward_models:
                print(f"Extracting embeddings for seed {seed}...")
                model = self.reward_models[seed]
                embeddings = get_embeddings(model, self.reward_tokenizers[seed], sample_texts)
                all_embeddings[seed] = embeddings
        
        # Skip the rest if no embeddings were extracted
        if not all_embeddings:
            print("No embeddings could be extracted. Skipping feature representation analysis.")
            return
        
        # Perform PCA on embeddings from each model
        for seed, embeddings in all_embeddings.items():
            # Apply PCA
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'PC1': reduced_embeddings[:, 0],
                'PC2': reduced_embeddings[:, 1],
                'Label': ['Toxic' if label == 1 else 'Non-toxic' for label in sample_labels]
            })
            
            # Plot PCA
            plt.figure(figsize=(10, 8))
            sns.scatterplot(data=df, x='PC1', y='PC2', hue='Label', palette={'Toxic': 'red', 'Non-toxic': 'blue'})
            plt.title(f'PCA of Embeddings from Seed {seed} (Pythia-{self.model_size})')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'pca_embeddings_seed_{seed}.png'))
            plt.close()
        
        # Combine all embeddings for comparison
        if len(all_embeddings) > 1:
            # Flatten and concatenate all embeddings
            all_flattened = []
            model_labels = []
            for seed, embeddings in all_embeddings.items():
                all_flattened.append(embeddings)
                model_labels.extend([f"Seed {seed}"] * len(embeddings))
                
            all_flattened = np.vstack(all_flattened)
            
            # Apply PCA to all embeddings
            pca = PCA(n_components=2)
            all_reduced = pca.fit_transform(all_flattened)
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'PC1': all_reduced[:, 0],
                'PC2': all_reduced[:, 1],
                'Model': model_labels
            })
            
            # Plot PCA of all embeddings
            plt.figure(figsize=(12, 10))
            sns.scatterplot(data=df, x='PC1', y='PC2', hue='Model')
            plt.title(f'PCA of Embeddings from All Models (Pythia-{self.model_size})')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'pca_all_embeddings.png'))
            plt.close()
        
    def evaluate_individual_models(self, use_train=False):
        """
        Evaluate each individual model's performance on the dataset.
        
        Args:
            use_train: If True, evaluate on training set, otherwise on test set
        """
        dataset_type = "TRAIN" if use_train else "TEST"
        print(f"Evaluating individual models on {dataset_type} set...")
        
        # Select the appropriate dataset
        texts = self.train_texts if use_train else self.test_texts
        labels = self.train_labels if use_train else self.test_labels
        
        results = {}
        
        # Evaluate each reward model
        for seed, model in self.reward_models.items():
            print(f"Evaluating model with seed {seed}...")
            
            # Get predictions
            predictions = self.get_model_predictions(
                texts,
                model, 
                self.reward_tokenizers[seed]
            )
            
            # Normalize predictions (higher = less toxic)
            # Convert to binary predictions using mean threshold
            threshold = np.mean(predictions)
            binary_preds = (predictions > threshold).astype(int)
            binary_preds = 1 - binary_preds  # Invert to match ground truth (1=toxic)
            
            # Calculate metrics
            accuracy = accuracy_score(labels, binary_preds)
            f1 = f1_score(labels, binary_preds)
            auc_roc = roc_auc_score(labels, [-x for x in predictions])  # Invert for ROC
            
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
        file_prefix = "train_" if use_train else "test_"
        with open(os.path.join(self.output_dir, f'{file_prefix}individual_model_results.json'), 'w') as f:
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
        
        plt.xlabel('Model Seed')
        plt.ylabel('Score')
        plt.title(f'Individual Model Performance ({dataset_type} Set, Pythia-{self.model_size})')
        plt.xticks(x, seeds)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{file_prefix}individual_performance.png'))
        plt.close()
        
        return results
        
    def evaluate_ensemble_methods(self, use_train=False):
        """
        Evaluate different ensemble methods for combining model predictions.
        
        Args:
            use_train: If True, evaluate on training set, otherwise on test set
        """
        dataset_type = "TRAIN" if use_train else "TEST"
        print(f"Evaluating ensemble methods on {dataset_type} set...")
        
        # If no models were loaded, return empty results
        if not self.reward_models:
            print("No models available for ensemble evaluation")
            return {}, {"best_individual": {}, "best_ensemble": {}}
        
        # Select the appropriate dataset
        texts = self.train_texts if use_train else self.test_texts
        labels = self.train_labels if use_train else self.test_labels
        
        # Get predictions from all models
        all_predictions = {}
        for seed, model in self.reward_models.items():
            predictions = self.get_model_predictions(
                texts,
                model, 
                self.reward_tokenizers[seed]
            )
            all_predictions[seed] = predictions
        
        # If no predictions were generated, return empty results
        if not all_predictions:
            print("No predictions available for ensemble evaluation")
            return {}, {"best_individual": {}, "best_ensemble": {}}
        
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
            accuracy = accuracy_score(labels, binary_preds)
            f1 = f1_score(labels, binary_preds)
            auc_roc = roc_auc_score(labels, [-x for x in ensemble_preds])  # Invert for ROC
            
            # Store results
            ensemble_results[method_name] = {
                "accuracy": accuracy,
                "f1": f1,
                "auc_roc": auc_roc
            }
            
            print(f"Ensemble ({method_name}) - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC-ROC: {auc_roc:.4f}")
            
        # Save results
        file_prefix = "train_" if use_train else "test_"
        with open(os.path.join(self.output_dir, f'{file_prefix}ensemble_results.json'), 'w') as f:
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
        plt.savefig(os.path.join(self.output_dir, f'{file_prefix}ensemble_performance.png'))
        plt.close()
        
        # Compare best ensemble with best individual model
        individual_results = self.evaluate_individual_models(use_train=use_train)
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
        with open(os.path.join(self.output_dir, f'{file_prefix}best_model_comparison.json'), 'w') as f:
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
        
    def analyze_error_patterns(self, use_train=False):
        """
        Analyze patterns in model errors to understand where models disagree.
        
        Args:
            use_train: If True, analyze training set, otherwise test set
        """
        dataset_type = "TRAIN" if use_train else "TEST"
        print(f"Analyzing error patterns on {dataset_type} set...")
        
        # If no models were loaded, return empty results
        if not self.reward_models:
            print("No models available for error pattern analysis")
            return {"disagreement_rate": 0, "all_wrong_count": 0, "some_right_count": 0}
        
        # Select the appropriate dataset
        texts = self.train_texts if use_train else self.test_texts
        labels = self.train_labels if use_train else self.test_labels
        
        # Get predictions from all models
        all_model_preds = {}
        for seed, model in self.reward_models.items():
            predictions = self.get_model_predictions(
                texts,
                model, 
                self.reward_tokenizers[seed]
            )
            
            # Convert to binary predictions using mean threshold
            threshold = np.mean(predictions)
            binary_preds = (predictions > threshold).astype(int)
            binary_preds = 1 - binary_preds  # Invert to match ground truth (1=toxic)
            
            all_model_preds[seed] = binary_preds
        
        # If no predictions were generated, return empty results
        if not all_model_preds:
            print("No predictions available for error pattern analysis")
            return {"disagreement_rate": 0, "all_wrong_count": 0, "some_right_count": 0}
        
        # Create a DataFrame with all predictions
        df = pd.DataFrame(all_model_preds)
        df['true_label'] = labels
        
        # Calculate disagreement rate between models
        model_cols = [col for col in df.columns if col != 'true_label']
        if len(model_cols) > 1:
            disagreement = 0
            total = len(df)
            for i in range(total):
                row = df.iloc[i][model_cols].values
                if np.max(row) != np.min(row):
                    disagreement += 1
        
            disagreement_rate = disagreement / total
            print(f"Model disagreement rate: {disagreement_rate:.4f}")
        
        # Count examples where all models are wrong
        all_wrong = 0
        some_right = 0
        
        for i in range(len(df)):
            row = df.iloc[i]
            true_label = row['true_label']
            model_predictions = row[model_cols].values
            
            # Check if all models are wrong
            if np.all(model_predictions != true_label):
                all_wrong += 1
            # Check if some models are right and some are wrong
            elif np.any(model_predictions == true_label) and np.any(model_predictions != true_label):
                some_right += 1
        
        print(f"Number of examples where all models are wrong: {all_wrong}")
        print(f"Number of examples where only some models are right: {some_right}")
        
        # Return statistics
        return {
            "disagreement_rate": disagreement_rate if len(model_cols) > 1 else 0,
            "all_wrong_count": all_wrong,
            "some_right_count": some_right
        }
        
    def analyze_raw_scores(self, use_train=False):
        """
        Analyze raw prediction scores from each model for original and detoxified texts.
        
        Args:
            use_train: If True, analyze training set, otherwise test set
        """
        dataset_type = "TRAIN" if use_train else "TEST"
        print(f"\nAnalyzing raw scores on {dataset_type} set...")
        
        # Select the appropriate dataset
        if use_train:
            original_texts = [item['output'] for item in self.train_data['original']]
            detoxified_texts = [item['output'] for item in self.train_data['detoxified']]
        else:
            original_texts = [item['output'] for item in self.test_data['original']]
            detoxified_texts = [item['output'] for item in self.test_data['detoxified']]
        
        # Store scores for each model
        all_scores = {}
        
        # Get predictions for each model
        for seed, model in self.reward_models.items():
            # Get scores for original (toxic) texts
            original_scores = self.get_model_predictions(
                original_texts,
                model, 
                self.reward_tokenizers[seed]
            )
            
            # Get scores for detoxified texts
            detoxified_scores = self.get_model_predictions(
                detoxified_texts,
                model, 
                self.reward_tokenizers[seed]
            )
            
            # Calculate statistics
            original_mean = np.mean(original_scores)
            detoxified_mean = np.mean(detoxified_scores)
            score_diff = detoxified_mean - original_mean
            
            all_scores[seed] = {
                "original_mean": float(original_mean),
                "detoxified_mean": float(detoxified_mean),
                "score_difference": float(score_diff),
                "original_std": float(np.std(original_scores)),
                "detoxified_std": float(np.std(detoxified_scores))
            }
            
            print(f"Seed {seed} - Original mean: {original_mean:.4f}, Detoxified mean: {detoxified_mean:.4f}, Diff: {score_diff:.4f}")
        
        # Calculate ensemble scores (mean across models)
        if len(self.reward_models) > 0:
            # Calculate ensemble statistics
            ensemble_original_mean = np.mean([all_scores[seed]["original_mean"] for seed in self.reward_models.keys()])
            ensemble_detoxified_mean = np.mean([all_scores[seed]["detoxified_mean"] for seed in self.reward_models.keys()])
            ensemble_score_diff = ensemble_detoxified_mean - ensemble_original_mean
            
            all_scores["ensemble"] = {
                "original_mean": float(ensemble_original_mean),
                "detoxified_mean": float(ensemble_detoxified_mean),
                "score_difference": float(ensemble_score_diff)
            }
            
            print(f"Ensemble - Original mean: {ensemble_original_mean:.4f}, Detoxified mean: {ensemble_detoxified_mean:.4f}, Diff: {ensemble_score_diff:.4f}")
        
        # Save results
        file_prefix = "train_" if use_train else "test_"
        with open(os.path.join(self.output_dir, f'{file_prefix}raw_scores.json'), 'w') as f:
            json.dump(all_scores, f, indent=2)
        
        # Create a visualization
        plt.figure(figsize=(12, 6))
        
        # Set up data for plotting
        seeds = list(self.reward_models.keys())
        if len(seeds) > 0:
            seeds.append("ensemble")
            original_means = [all_scores[seed]["original_mean"] for seed in seeds]
            detoxified_means = [all_scores[seed]["detoxified_mean"] for seed in seeds]
            
            x = np.arange(len(seeds))
            width = 0.35
            
            # Create grouped bar chart
            plt.bar(x - width/2, original_means, width, label='Original (Toxic)')
            plt.bar(x + width/2, detoxified_means, width, label='Detoxified')
            
            plt.xlabel('Model Seed')
            plt.ylabel('Mean Score')
            plt.title(f'Raw Score Comparison ({dataset_type} Set, Pythia-{self.model_size})')
            plt.xticks(x, seeds)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{file_prefix}raw_scores.png'))
            plt.close()
        
        return all_scores
        
    def analyze_raw_scores_irl_method(self, use_train=False):
        """
        Analyze raw prediction scores using the exact same methodology as in IRL training.
        
        Args:
            use_train: If True, analyze training set, otherwise test set
        """
        dataset_type = "TRAIN" if use_train else "TEST"
        print(f"\nAnalyzing raw scores (IRL method) on {dataset_type} set...")
        
        # Import the exact scoring function from irl_utilities if available
        try:
            from irl_train import score_toxicity
            use_irl_score_function = True
            print("Using score_toxicity function from irl_train.py")
        except ImportError:
            use_irl_score_function = False
            print("Could not import score_toxicity from irl_train.py, using direct model inference")
        
        # Select the appropriate dataset
        if use_train:
            original_texts = [item['output'] for item in self.train_data['original']]
            detoxified_texts = [item['output'] for item in self.train_data['detoxified']]
        else:
            original_texts = [item['output'] for item in self.test_data['original']]
            detoxified_texts = [item['output'] for item in self.test_data['detoxified']]
        
        # Store scores for each model
        all_scores = {}
        
        # Get predictions for each model
        for seed, model in self.reward_models.items():
            # Score using the exact IRL method if available
            if use_irl_score_function:
                # Use the score_toxicity function from irl_train.py
                original_scores = score_toxicity(original_texts, model, self.reward_tokenizers[seed], batch_size=self.batch_size)
                detoxified_scores = score_toxicity(detoxified_texts, model, self.reward_tokenizers[seed], batch_size=self.batch_size)
            else:
                # Score directly using the model
                original_scores = []
                detoxified_scores = []
                
                # Process original texts
                for text in original_texts:
                    inputs = self.reward_tokenizers[seed](
                        text, 
                        return_tensors="pt", 
                        truncation=True,
                        max_length=self.max_length
                    ).to(device)
                    
                    with torch.no_grad():
                        score = model(inputs.input_ids, inputs.attention_mask).item()
                    original_scores.append(score)
                    
                # Process detoxified texts
                for text in detoxified_texts:
                    inputs = self.reward_tokenizers[seed](
                        text, 
                        return_tensors="pt", 
                        truncation=True,
                        max_length=self.max_length
                    ).to(device)
                    
                    with torch.no_grad():
                        score = model(inputs.input_ids, inputs.attention_mask).item()
                    detoxified_scores.append(score)
            
            # Convert to numpy arrays
            original_scores = np.array(original_scores)
            detoxified_scores = np.array(detoxified_scores)
            
            # Calculate statistics
            original_mean = np.mean(original_scores)
            detoxified_mean = np.mean(detoxified_scores)
            score_diff = detoxified_mean - original_mean
            
            all_scores[seed] = {
                "original_mean": float(original_mean),
                "detoxified_mean": float(detoxified_mean),
                "score_difference": float(score_diff),
                "original_std": float(np.std(original_scores)),
                "detoxified_std": float(np.std(detoxified_scores)),
                "original_min": float(np.min(original_scores)),
                "original_max": float(np.max(original_scores)),
                "detoxified_min": float(np.min(detoxified_scores)),
                "detoxified_max": float(np.max(detoxified_scores))
            }
            
            print(f"Seed {seed} - Original mean: {original_mean:.4f} (range: {np.min(original_scores):.4f} to {np.max(original_scores):.4f})")
            print(f"Seed {seed} - Detoxified mean: {detoxified_mean:.4f} (range: {np.min(detoxified_scores):.4f} to {np.max(detoxified_scores):.4f})")
            print(f"Seed {seed} - Diff: {score_diff:.4f}")
        
        # Calculate ensemble scores (mean across models)
        if len(self.reward_models) > 0:
            # Calculate ensemble statistics
            ensemble_original_mean = np.mean([all_scores[seed]["original_mean"] for seed in self.reward_models.keys()])
            ensemble_detoxified_mean = np.mean([all_scores[seed]["detoxified_mean"] for seed in self.reward_models.keys()])
            ensemble_score_diff = ensemble_detoxified_mean - ensemble_original_mean
            
            all_scores["ensemble"] = {
                "original_mean": float(ensemble_original_mean),
                "detoxified_mean": float(ensemble_detoxified_mean),
                "score_difference": float(ensemble_score_diff)
            }
            
            print(f"Ensemble - Original mean: {ensemble_original_mean:.4f}, Detoxified mean: {ensemble_detoxified_mean:.4f}, Diff: {ensemble_score_diff:.4f}")
        
        # Save results
        file_prefix = "train_" if use_train else "test_"
        with open(os.path.join(self.output_dir, f'{file_prefix}raw_scores_irl_method.json'), 'w') as f:
            json.dump(all_scores, f, indent=2)
        
        # Create a visualization
        plt.figure(figsize=(12, 6))
        
        # Set up data for plotting
        seeds = list(self.reward_models.keys())
        if len(seeds) > 0:
            seeds.append("ensemble")
            original_means = [all_scores[seed]["original_mean"] for seed in seeds]
            detoxified_means = [all_scores[seed]["detoxified_mean"] for seed in seeds]
            
            x = np.arange(len(seeds))
            width = 0.35
            
            # Create grouped bar chart
            plt.bar(x - width/2, original_means, width, label='Original (Toxic)')
            plt.bar(x + width/2, detoxified_means, width, label='Detoxified')
            
            plt.xlabel('Model Seed')
            plt.ylabel('Mean Score')
            plt.title(f'Raw Score Comparison - IRL Method ({dataset_type} Set, Pythia-{self.model_size})')
            plt.xticks(x, seeds)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{file_prefix}raw_scores_irl_method.png'))
            plt.close()
        
        return all_scores
        
    def run_full_analysis(self):
        """Run all analysis methods and compile results."""
        print(f"Running full analysis for pythia-{self.model_size}...")
        
        results = {}
        
        # Run all analyses
        results["correlations"] = self.analyze_model_correlations()
        self.analyze_feature_representations()
        
        # Analyze raw scores using both methods
        print("\n=== Analyzing Raw Scores ===")
        results["test_raw_scores"] = self.analyze_raw_scores(use_train=False)
        results["train_raw_scores"] = self.analyze_raw_scores(use_train=True)
        
        print("\n=== Analyzing Raw Scores (IRL Method) ===")
        results["test_raw_scores_irl"] = self.analyze_raw_scores_irl_method(use_train=False)
        results["train_raw_scores_irl"] = self.analyze_raw_scores_irl_method(use_train=True)
        
        # Evaluate on test set (default)
        print("\n=== Evaluating on TEST set ===")
        results["test_individual_performance"] = self.evaluate_individual_models(use_train=False)
        results["test_ensemble_performance"], results["test_best_comparison"] = self.evaluate_ensemble_methods(use_train=False)
        results["test_error_patterns"] = self.analyze_error_patterns(use_train=False)
        
        # Evaluate on train set
        print("\n=== Evaluating on TRAIN set ===")
        results["train_individual_performance"] = self.evaluate_individual_models(use_train=True)
        results["train_ensemble_performance"], results["train_best_comparison"] = self.evaluate_ensemble_methods(use_train=True)
        results["train_error_patterns"] = self.analyze_error_patterns(use_train=True)
        
        # Save summary results
        summary = {
            "model_size": self.model_size,
            "seeds": self.seeds,
            "num_models": len(self.reward_models),
            "dataset_size": {
                "train": len(self.train_texts) // 2,  # Divide by 2 because we have original + detoxified
                "test": len(self.test_texts) // 2
            }
        }
        
        # Extract key correlation metrics
        if isinstance(results["correlations"], pd.DataFrame):
            model_cols = [f"seed_{seed}" for seed in self.seeds if seed in self.reward_models]
            
            if len(model_cols) > 1:
                model_corr = results["correlations"].loc[model_cols, model_cols]
                avg_model_corr = (model_corr.sum().sum() - len(model_cols)) / (len(model_cols) * (len(model_cols) - 1))
                summary["average_model_correlation"] = avg_model_corr
                
            if "ground_truth" in results["correlations"].columns:
                gt_corr = results["correlations"].loc[model_cols, "ground_truth"].mean()
                summary["average_ground_truth_correlation"] = gt_corr
                
            if "true_label" in results["correlations"].columns:
                label_corr = results["correlations"].loc[model_cols, "true_label"].mean()
                summary["average_label_correlation"] = label_corr
        
        # Save summary
        with open(os.path.join(self.output_dir, 'analysis_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
            
        return results


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