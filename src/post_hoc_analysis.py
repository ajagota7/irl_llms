#!/usr/bin/env python
"""
Post-hoc analysis script for IRL experiments.
Analyzes reward model performance at granular levels and compares across different experimental conditions.
"""

import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.irl_utilities import RewardModel, plot_score_distribution


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    model_name: str
    model_size: str  # e.g., "70m", "410m", "1b", "llama-1b"
    toxicity_threshold: float  # 0.3 or 0.8
    input_type: str  # "prompt_output" or "output_only"
    reward_model_id: str
    original_dataset_id: str
    detoxified_dataset_id: str


class PostHocAnalyzer:
    """Main class for post-hoc analysis of IRL experiments."""
    
    def __init__(self, device: str = None, seed: int = 42, train_test_split: float = 0.8):
        """Initialize the analyzer."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_test_split = train_test_split
        self.set_seeds()
        
        # Store loaded models and datasets
        self.reward_models = {}
        self.tokenizers = {}
        self.datasets = {}
        
        # Analysis results
        self.results = {}
        
    def set_seeds(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
    
    def load_reward_model(self, model_id: str) -> Tuple[RewardModel, AutoTokenizer]:
        """Load a reward model and tokenizer."""
        if model_id in self.reward_models:
            return self.reward_models[model_id], self.tokenizers[model_id]
        
        print(f"Loading reward model: {model_id}")
        
        try:
            # Create reward model
            reward_model = RewardModel(
                model_name=model_id,
                device=self.device,
                num_unfrozen_layers=0  # For inference
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
            
            # Store for reuse
            self.reward_models[model_id] = reward_model
            self.tokenizers[model_id] = tokenizer
            
            return reward_model, tokenizer
            
        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            raise
    
    def load_dataset(self, dataset_id: str) -> List[Dict]:
        """Load a dataset from HuggingFace."""
        if dataset_id in self.datasets:
            return self.datasets[dataset_id]
        
        print(f"Loading dataset: {dataset_id}")
        
        try:
            from datasets import load_dataset
            
            # Try multiple loading strategies
            try:
                dataset = load_dataset(dataset_id)
                if isinstance(dataset, dict) and 'train' in dataset:
                    data = dataset['train']
                else:
                    data = dataset
                
                # Convert to list of dictionaries
                if hasattr(data, 'to_pandas'):
                    data = data.to_pandas().to_dict('records')
                else:
                    data = [item for item in data]
                    
            except Exception as e:
                print(f"Error with default loading: {e}")
                # Try with streaming
                dataset = load_dataset(dataset_id, streaming=True)
                if isinstance(dataset, dict) and 'train' in dataset:
                    data = list(dataset['train'].take(2000))
                else:
                    data = list(dataset.take(2000))
            
            self.datasets[dataset_id] = data
            return data
            
        except Exception as e:
            print(f"Error loading dataset {dataset_id}: {e}")
            # Return dummy data for testing
            return [{"output": f"Dummy text {i}"} for i in range(2000)]
    
    def score_texts(self, texts: List[str], reward_model: RewardModel, 
                   tokenizer: AutoTokenizer, batch_size: int = 8, 
                   max_length: int = 512) -> List[float]:
        """Score a list of texts using the reward model."""
        reward_model.eval()
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length
                )
                
                # Move to device
                inputs = {k: v.to(reward_model.device) for k, v in inputs.items()}
                
                # Get rewards
                rewards = reward_model(**inputs)
                
                # Convert to list of floats
                rewards_list = rewards.squeeze().cpu().tolist()
                if not isinstance(rewards_list, list):
                    rewards_list = [rewards_list]
                
                all_scores.extend(rewards_list)
        
        return all_scores
    
    def prepare_train_test_split(self, original_data: List[Dict], 
                               detoxified_data: List[Dict]) -> Tuple[Dict, Dict]:
        """Prepare train/test split of the data."""
        assert len(original_data) == len(detoxified_data), "Dataset lengths must match"
        
        # Set random seed for reproducible split
        np.random.seed(self.seed)
        
        # Create indices and shuffle
        indices = np.arange(len(original_data))
        np.random.shuffle(indices)
        
        # Split data
        train_size = int(self.train_test_split * len(original_data))
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        train_data = {
            'original': [original_data[i] for i in train_indices],
            'detoxified': [detoxified_data[i] for i in train_indices]
        }
        
        test_data = {
            'original': [original_data[i] for i in test_indices],
            'detoxified': [detoxified_data[i] for i in test_indices]
        }
        
        print(f"Train set: {len(train_data['original'])} samples")
        print(f"Test set: {len(test_data['original'])} samples")
        
        return train_data, test_data
    
    def analyze_single_experiment(self, config: ExperimentConfig, 
                                batch_size: int = 8, max_length: int = 512) -> Dict:
        """Analyze a single experiment in detail with train/test split."""
        print(f"\nAnalyzing experiment: {config.model_size} - {config.input_type} - tox{config.toxicity_threshold}")
        
        # Load model and datasets
        reward_model, tokenizer = self.load_reward_model(config.reward_model_id)
        original_data = self.load_dataset(config.original_dataset_id)
        detoxified_data = self.load_dataset(config.detoxified_dataset_id)
        
        # Prepare train/test split
        train_data, test_data = self.prepare_train_test_split(original_data, detoxified_data)
        
        results = {
            'config': config,
            'train': self._analyze_split(train_data, reward_model, tokenizer, batch_size, max_length, "train"),
            'test': self._analyze_split(test_data, reward_model, tokenizer, batch_size, max_length, "test")
        }
        
        return results
    
    def _analyze_split(self, data: Dict, reward_model: RewardModel, 
                      tokenizer: AutoTokenizer, batch_size: int, max_length: int, 
                      split_name: str) -> Dict:
        """Analyze a single data split (train or test)."""
        # Extract texts
        original_texts = [item.get('output', item.get('text', '')) for item in data['original']]
        detoxified_texts = [item.get('output', item.get('text', '')) for item in data['detoxified']]
        
        # Score texts
        original_scores = self.score_texts(original_texts, reward_model, tokenizer, batch_size, max_length)
        detoxified_scores = self.score_texts(detoxified_texts, reward_model, tokenizer, batch_size, max_length)
        
        # Calculate improvements
        improvements = [detox - orig for orig, detox in zip(original_scores, detoxified_scores)]
        relative_improvements = [(detox - orig) / abs(orig) if orig != 0 else 0 
                               for orig, detox in zip(original_scores, detoxified_scores)]
        
        # Create detailed results
        results = {
            'original_scores': original_scores,
            'detoxified_scores': detoxified_scores,
            'improvements': improvements,
            'relative_improvements': relative_improvements,
            'original_texts': original_texts,
            'detoxified_texts': detoxified_texts,
            
            # Summary statistics
            'summary': {
                'mean_original_score': np.mean(original_scores),
                'mean_detoxified_score': np.mean(detoxified_scores),
                'mean_improvement': np.mean(improvements),
                'mean_relative_improvement': np.mean(relative_improvements),
                'std_original_score': np.std(original_scores),
                'std_detoxified_score': np.std(detoxified_scores),
                'std_improvement': np.std(improvements),
                'min_improvement': np.min(improvements),
                'max_improvement': np.max(improvements),
                'positive_improvements': sum(1 for imp in improvements if imp > 0),
                'total_samples': len(improvements),
                'improvement_rate': sum(1 for imp in improvements if imp > 0) / len(improvements),
                'median_improvement': np.median(improvements),
                'q25_improvement': np.percentile(improvements, 25),
                'q75_improvement': np.percentile(improvements, 75)
            }
        }
        
        # Find top and bottom performers
        sorted_indices = np.argsort(improvements)[::-1]  # Descending order
        
        results['top_improvers'] = [
            {
                'index': int(idx),
                'original_score': original_scores[idx],
                'detoxified_score': detoxified_scores[idx],
                'improvement': improvements[idx],
                'relative_improvement': relative_improvements[idx],
                'original_text': original_texts[idx],
                'detoxified_text': detoxified_texts[idx]
            }
            for idx in sorted_indices[:20]  # Top 20
        ]
        
        results['bottom_improvers'] = [
            {
                'index': int(idx),
                'original_score': original_scores[idx],
                'detoxified_score': detoxified_scores[idx],
                'improvement': improvements[idx],
                'relative_improvement': relative_improvements[idx],
                'original_text': original_texts[idx],
                'detoxified_text': detoxified_texts[idx]
            }
            for idx in sorted_indices[-20:]  # Bottom 20
        ]
        
        # Create full sample dataframe
        sample_data = []
        for i in range(len(original_texts)):
            sample_data.append({
                'sample_index': i,
                'original_score': original_scores[i],
                'detoxified_score': detoxified_scores[i],
                'improvement': improvements[i],
                'relative_improvement': relative_improvements[i],
                'original_text': original_texts[i],
                'detoxified_text': detoxified_texts[i],
                'improvement_rank': np.where(sorted_indices == i)[0][0] + 1
            })
        
        results['sample_dataframe'] = pd.DataFrame(sample_data)
        
        return results
    
    def compare_experiments(self, configs: List[ExperimentConfig], 
                          comparison_type: str) -> Dict:
        """Compare multiple experiments."""
        print(f"\nComparing experiments: {comparison_type}")
        
        # Analyze each experiment
        experiment_results = {}
        for config in configs:
            key = f"{config.model_size}_{config.input_type}_{config.toxicity_threshold}"
            experiment_results[key] = self.analyze_single_experiment(config)
        
        # Create comparison dataframe
        comparison_data = []
        for key, results in experiment_results.items():
            for split in ['train', 'test']:
                summary = results[split]['summary']
                config = results['config']
                
                comparison_data.append({
                    'experiment_key': key,
                    'split': split,
                    'model_size': config.model_size,
                    'input_type': config.input_type,
                    'toxicity_threshold': config.toxicity_threshold,
                    'mean_original_score': summary['mean_original_score'],
                    'mean_detoxified_score': summary['mean_detoxified_score'],
                    'mean_improvement': summary['mean_improvement'],
                    'mean_relative_improvement': summary['mean_relative_improvement'],
                    'improvement_rate': summary['improvement_rate'],
                    'std_improvement': summary['std_improvement'],
                    'max_improvement': summary['max_improvement'],
                    'min_improvement': summary['min_improvement'],
                    'median_improvement': summary['median_improvement'],
                    'q25_improvement': summary['q25_improvement'],
                    'q75_improvement': summary['q75_improvement']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        return {
            'comparison_type': comparison_type,
            'experiment_results': experiment_results,
            'comparison_dataframe': comparison_df
        }
    
    def analyze_cross_model_patterns(self, all_configs: List[ExperimentConfig]) -> Dict:
        """Analyze patterns across model sizes within experiment sets."""
        print("\n=== Analyzing Cross-Model Patterns ===")
        
        # Group experiments by input type and toxicity threshold
        experiment_sets = {}
        for config in all_configs:
            key = f"{config.input_type}_tox{config.toxicity_threshold}"
            if key not in experiment_sets:
                experiment_sets[key] = []
            experiment_sets[key].append(config)
        
        pattern_results = {}
        
        for set_key, configs in experiment_sets.items():
            print(f"\nAnalyzing pattern set: {set_key}")
            
            # Analyze all experiments in this set
            experiment_results = {}
            for config in configs:
                key = f"{config.model_size}_{config.input_type}_{config.toxicity_threshold}"
                experiment_results[key] = self.analyze_single_experiment(config)
            
            # Find consistent improvers across model sizes
            consistent_improvers = self._find_consistent_improvers_across_models(
                experiment_results, set_key
            )
            
            # Analyze improvement patterns
            improvement_patterns = self._analyze_improvement_patterns(
                experiment_results, set_key
            )
            
            pattern_results[set_key] = {
                'experiment_results': experiment_results,
                'consistent_improvers': consistent_improvers,
                'improvement_patterns': improvement_patterns
            }
        
        return pattern_results
    
    def _find_consistent_improvers_across_models(self, experiment_results: Dict, 
                                               set_key: str) -> Dict:
        """Find samples that consistently improve across model sizes within a set."""
        # Get all test splits (for consistency)
        test_splits = {}
        for exp_key, results in experiment_results.items():
            test_splits[exp_key] = results['test']
        
        # Create a matrix of improvements for each sample across models
        sample_improvements = {}
        
        for exp_key, test_results in test_splits.items():
            improvements = test_results['improvements']
            for i, improvement in enumerate(improvements):
                if i not in sample_improvements:
                    sample_improvements[i] = {}
                sample_improvements[i][exp_key] = improvement
        
        # Find samples that improve in most models
        consistent_improvers = []
        for sample_idx, improvements in sample_improvements.items():
            positive_count = sum(1 for imp in improvements.values() if imp > 0)
            total_models = len(improvements)
            improvement_rate = positive_count / total_models
            
            if improvement_rate >= 0.75:  # Improve in at least 75% of models
                consistent_improvers.append({
                    'sample_index': sample_idx,
                    'improvement_rate': improvement_rate,
                    'improvements': improvements,
                    'mean_improvement': np.mean(list(improvements.values())),
                    'std_improvement': np.std(list(improvements.values())),
                    'model_consistency': {model: imp > 0 for model, imp in improvements.items()}
                })
        
        # Sort by mean improvement
        consistent_improvers.sort(key=lambda x: x['mean_improvement'], reverse=True)
        
        return {
            'consistent_improvers': consistent_improvers,
            'total_consistent_improvers': len(consistent_improvers),
            'total_samples': len(sample_improvements)
        }
    
    def _analyze_improvement_patterns(self, experiment_results: Dict, set_key: str) -> Dict:
        """Analyze patterns in improvements across model sizes."""
        # Get test splits
        test_splits = {}
        for exp_key, results in experiment_results.items():
            test_splits[exp_key] = results['test']
        
        # Analyze correlation between model size and improvement
        model_sizes = []
        mean_improvements = []
        
        for exp_key, test_results in test_splits.items():
            # Extract model size from key
            model_size = exp_key.split('_')[0]
            model_sizes.append(model_size)
            mean_improvements.append(test_results['summary']['mean_improvement'])
        
        # Create pattern analysis
        pattern_analysis = {
            'model_sizes': model_sizes,
            'mean_improvements': mean_improvements,
            'improvement_trend': 'increasing' if len(mean_improvements) > 1 and mean_improvements[-1] > mean_improvements[0] else 'decreasing' if len(mean_improvements) > 1 and mean_improvements[-1] < mean_improvements[0] else 'stable'
        }
        
        return pattern_analysis
    
    def find_consistent_improvers(self, all_configs: List[ExperimentConfig]) -> Dict:
        """Find samples that consistently improve across all experiments."""
        print("\nFinding consistent improvers across all experiments...")
        
        # Analyze all experiments
        all_results = {}
        for config in all_configs:
            key = f"{config.model_size}_{config.input_type}_{config.toxicity_threshold}"
            all_results[key] = self.analyze_single_experiment(config)
        
        # Create a matrix of improvements for each sample across experiments
        sample_improvements = {}
        
        for exp_key, results in all_results.items():
            # Use test split for consistency
            test_results = results['test']
            improvements = test_results['improvements']
            for i, improvement in enumerate(improvements):
                if i not in sample_improvements:
                    sample_improvements[i] = {}
                sample_improvements[i][exp_key] = improvement
        
        # Find samples that improve in most experiments
        consistent_improvers = []
        for sample_idx, improvements in sample_improvements.items():
            positive_count = sum(1 for imp in improvements.values() if imp > 0)
            total_experiments = len(improvements)
            improvement_rate = positive_count / total_experiments
            
            if improvement_rate >= 0.75:  # Improve in at least 75% of experiments
                consistent_improvers.append({
                    'sample_index': sample_idx,
                    'improvement_rate': improvement_rate,
                    'improvements': improvements,
                    'mean_improvement': np.mean(list(improvements.values())),
                    'std_improvement': np.std(list(improvements.values()))
                })
        
        # Sort by mean improvement
        consistent_improvers.sort(key=lambda x: x['mean_improvement'], reverse=True)
        
        return {
            'consistent_improvers': consistent_improvers,
            'total_consistent_improvers': len(consistent_improvers),
            'all_results': all_results
        }
    
    def save_results(self, results: Dict, output_dir: str):
        """Save analysis results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        results_file = os.path.join(output_dir, f"post_hoc_analysis_{timestamp}.json")
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._convert_to_json_serializable(results)
            json.dump(json_results, f, indent=2)
        
        # Save individual experiment CSVs
        for key, result in results.items():
            if key.startswith('individual_'):
                exp_name = key.replace('individual_', '')
                
                # Save train split
                train_df = result['train']['sample_dataframe']
                train_csv = os.path.join(output_dir, f"{exp_name}_train_samples_{timestamp}.csv")
                train_df.to_csv(train_csv, index=False)
                
                # Save test split
                test_df = result['test']['sample_dataframe']
                test_csv = os.path.join(output_dir, f"{exp_name}_test_samples_{timestamp}.csv")
                test_df.to_csv(test_csv, index=False)
        
        # Save comparison CSVs
        for key, result in results.items():
            if 'comparison_dataframe' in result:
                comparison_name = key.replace('_comparison', '')
                csv_file = os.path.join(output_dir, f"{comparison_name}_comparison_{timestamp}.csv")
                result['comparison_dataframe'].to_csv(csv_file, index=False)
        
        # Save cross-model pattern results
        if 'cross_model_patterns' in results:
            for set_key, pattern_result in results['cross_model_patterns'].items():
                # Save consistent improvers
                if 'consistent_improvers' in pattern_result:
                    consistent_df = pd.DataFrame(pattern_result['consistent_improvers'])
                    consistent_csv = os.path.join(output_dir, f"{set_key}_consistent_improvers_{timestamp}.csv")
                    consistent_df.to_csv(consistent_csv, index=False)
        
        print(f"Results saved to {output_dir}")
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj


def get_experiment_configs() -> List[ExperimentConfig]:
    """Define all experiment configurations."""
    configs = []
    
    # Model configurations
    models = [
        ("70m", "EleutherAI/pythia-70m"),
        ("410m", "EleutherAI/pythia-410m"),
        ("1b", "EleutherAI/pythia-1b"),
        ("llama-1b", "meta-llama/Llama-3.2-1B")
    ]
    
    # Toxicity thresholds
    toxicity_levels = [0.3, 0.8]
    
    # Input types
    input_types = ["prompt_output", "output_only"]
    
    # Dataset and model mappings
    dataset_mappings = {
        # Tox 0.3 datasets
        ("70m", 0.3): (
            "ajagota71/EleutherAI_pythia-70M_2000_samples_original",
            "ajagota71/ajagota71_pythia-70m-detox-epoch-100_2000_samples_detoxified"
        ),
        ("410m", 0.3): (
            "ajagota71/EleutherAI_pythia-410m_2000_samples_temp0p7_tox0p3_original",
            "ajagota71/ajagota71_pythia-410m-s-nlp-detox-checkpoint-epoch-100_2000_samples_temp0p7_tox0p3_detoxified"
        ),
        ("1b", 0.3): (
            "ajagota71/EleutherAI_pythia-1b_2000_samples_temp0p7_tox0p3_original",
            "ajagota71/ajagota71_pythia-1b-s-nlp-detox-checkpoint-epoch-100_2000_samples_temp0p7_tox0p3_detoxified"
        ),
        ("llama-1b", 0.3): (
            "ajagota71/meta-llama_Llama-3.2-1B_2000_samples_temp0p7_tox0p3_original",
            "ajagota71/ajagota71_llama-3-2-1b-rlhf-kl-p5-target-2p5-lr-3e-6_2000_samples_temp0p7_tox0p3_detoxified"
        ),
        
        # Tox 0.8 datasets
        ("70m", 0.8): (
            "ajagota71/EleutherAI_pythia-70m_2000_samples_temp0p7_tox0p8_original",
            "ajagota71/ajagota71_pythia-70m-s-nlp-detox-checkpoint-epoch-100_2000_samples_temp0p7_tox0p8_detoxified"
        ),
        ("410m", 0.8): (
            "ajagota71/EleutherAI_pythia-410m_2000_samples_temp0p7_tox0p8_original",
            "ajagota71/ajagota71_pythia-410m-s-nlp-detox-checkpoint-epoch-100_2000_samples_temp0p7_tox0p8_detoxified"
        ),
        ("1b", 0.8): (
            "ajagota71/EleutherAI_pythia-1b_2000_samples_temp0p7_tox0p8_original",
            "ajagota71/ajagota71_pythia-1b-s-nlp-detox-checkpoint-epoch-100_2000_samples_temp0p7_tox0p8_detoxified"
        ),
        ("llama-1b", 0.8): (
            "ajagota71/meta-llama_Llama-3.2-1B_2000_samples_temp0p7_tox0p8_original",
            "ajagota71/ajagota71_llama-3-2-1b-rlhf-kl-p5-target-2p5-lr-3e-6_2000_samples_temp0p7_tox0p8_detoxified"
        )
    }
    
    # Reward model mappings
    reward_model_mappings = {
        # Prompt + Output - Tox 0.3
        ("70m", "prompt_output", 0.3): "ajagota71/toxicity-reward-model-v-head-prompt-output-max-margin-seed-42-pythia-70m",
        ("410m", "prompt_output", 0.3): "ajagota71/toxicity-reward-model-v-head-prompt-output-max-margin-seed-42-pythia-410m",
        ("1b", "prompt_output", 0.3): "ajagota71/toxicity-reward-model-v-head-prompt-output-max-margin-seed-42-pythia-1b",
        ("llama-1b", "prompt_output", 0.3): "ajagota71/toxicity-reward-model-v-head-prompt-output-max-margin-seed-42-llama-3.2-1b",
        
        # Prompt + Output - Tox 0.8
        ("70m", "prompt_output", 0.8): "ajagota71/toxicity-reward-model-p8-v-head-prompt-output-max-margin-seed-42-pythia-70m",
        ("410m", "prompt_output", 0.8): "ajagota71/toxicity-reward-model-p8-v-head-prompt-output-max-margin-seed-42-pythia-410m",
        ("1b", "prompt_output", 0.8): "ajagota71/toxicity-reward-model-p8-v-head-prompt-output-max-margin-seed-42-pythia-1b",
        ("llama-1b", "prompt_output", 0.8): "ajagota71/toxicity-reward-model-p8-v-head-prompt-output-max-margin-seed-42-llama-3.2-1b",
        
        # Output Only - Tox 0.3
        ("70m", "output_only", 0.3): "ajagota71/toxicity-reward-model-v-head-output-max-margin-seed-42-pythia-70m",
        ("410m", "output_only", 0.3): "ajagota71/toxicity-reward-model-v-head-output-max-margin-seed-42-pythia-410m",
        ("1b", "output_only", 0.3): "ajagota71/toxicity-reward-model-v-head-output-max-margin-seed-42-pythia-1b",
        ("llama-1b", "output_only", 0.3): "ajagota71/toxicity-reward-model-v-head-output-max-margin-seed-42-llama-3.2-1b",
        
        # Output Only - Tox 0.8
        ("70m", "output_only", 0.8): "ajagota71/toxicity-reward-model-p8-v-head-output-max-margin-seed-42-pythia-70m",
        ("410m", "output_only", 0.8): "ajagota71/toxicity-reward-model-p8-v-head-output-max-margin-seed-42-pythia-410m",
        ("1b", "output_only", 0.8): "ajagota71/toxicity-reward-model-p8-v-head-output-max-margin-seed-42-pythia-1b",
        ("llama-1b", "output_only", 0.8): "ajagota71/toxicity-reward-model-p8-v-head-output-max-margin-seed-42-llama-3.2-1b"
    }
    
    # Generate all configurations
    for model_size, base_model in models:
        for toxicity_threshold in toxicity_levels:
            for input_type in input_types:
                # Get dataset IDs
                original_dataset_id, detoxified_dataset_id = dataset_mappings[(model_size, toxicity_threshold)]
                
                # Get reward model ID
                reward_model_id = reward_model_mappings[(model_size, input_type, toxicity_threshold)]
                
                config = ExperimentConfig(
                    model_name=base_model,
                    model_size=model_size,
                    toxicity_threshold=toxicity_threshold,
                    input_type=input_type,
                    reward_model_id=reward_model_id,
                    original_dataset_id=original_dataset_id,
                    detoxified_dataset_id=detoxified_dataset_id
                )
                configs.append(config)
    
    return configs


def main():
    parser = argparse.ArgumentParser(description="Post-hoc analysis of IRL experiments")
    
    parser.add_argument("--output_dir", type=str, default="post_hoc_analysis",
                        help="Output directory for analysis results")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run models on (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for processing")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum token length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--train_test_split", type=float, default=0.8,
                        help="Train/test split ratio (default: 0.8 for 1600/400 split)")
    
    # Analysis types
    parser.add_argument("--analyze_individual", action="store_true",
                        help="Analyze individual experiments in detail")
    parser.add_argument("--compare_input_types", action="store_true",
                        help="Compare prompt+output vs output-only")
    parser.add_argument("--compare_toxicity", action="store_true",
                        help="Compare low vs high toxicity thresholds")
    parser.add_argument("--analyze_cross_model_patterns", action="store_true",
                        help="Analyze patterns across model sizes within experiment sets")
    parser.add_argument("--find_consistent", action="store_true",
                        help="Find samples that consistently improve across experiments")
    parser.add_argument("--analyze_all", action="store_true",
                        help="Run all analyses")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = PostHocAnalyzer(device=args.device, seed=args.seed, train_test_split=args.train_test_split)
    
    # Get all experiment configurations
    all_configs = get_experiment_configs()
    print(f"Loaded {len(all_configs)} experiment configurations")
    
    results = {}
    
    # Individual experiment analysis
    if args.analyze_individual or args.analyze_all:
        print("\n=== Analyzing Individual Experiments ===")
        for config in all_configs:
            key = f"{config.model_size}_{config.input_type}_{config.toxicity_threshold}"
            results[f"individual_{key}"] = analyzer.analyze_single_experiment(
                config, args.batch_size, args.max_length
            )
    
    # Compare input types (prompt+output vs output-only)
    if args.compare_input_types or args.analyze_all:
        print("\n=== Comparing Input Types ===")
        for model_size in ["70m", "410m", "1b", "llama-1b"]:
            for toxicity in [0.3, 0.8]:
                configs = [config for config in all_configs 
                          if config.model_size == model_size and config.toxicity_threshold == toxicity]
                if len(configs) == 2:  # Should have both input types
                    comparison_key = f"input_type_comparison_{model_size}_tox{toxicity}"
                    results[comparison_key] = analyzer.compare_experiments(
                        configs, f"input_type_{model_size}_tox{toxicity}"
                    )
    
    # Compare toxicity thresholds
    if args.compare_toxicity or args.analyze_all:
        print("\n=== Comparing Toxicity Thresholds ===")
        for model_size in ["70m", "410m", "1b", "llama-1b"]:
            for input_type in ["prompt_output", "output_only"]:
                configs = [config for config in all_configs 
                          if config.model_size == model_size and config.input_type == input_type]
                if len(configs) == 2:  # Should have both toxicity levels
                    comparison_key = f"toxicity_comparison_{model_size}_{input_type}"
                    results[comparison_key] = analyzer.compare_experiments(
                        configs, f"toxicity_{model_size}_{input_type}"
                    )
    
    # Analyze cross-model patterns
    if args.analyze_cross_model_patterns or args.analyze_all:
        print("\n=== Analyzing Cross-Model Patterns ===")
        results["cross_model_patterns"] = analyzer.analyze_cross_model_patterns(all_configs)
    
    # Find consistent improvers
    if args.find_consistent or args.analyze_all:
        print("\n=== Finding Consistent Improvers ===")
        results["consistent_improvers"] = analyzer.find_consistent_improvers(all_configs)
    
    # Save results
    analyzer.save_results(results, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 