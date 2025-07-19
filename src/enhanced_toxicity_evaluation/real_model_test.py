#!/usr/bin/env python3
"""
Real end-to-end test with actual model generations and classifiers.
Uses base model and detoxified model to generate outputs, then classifies everything.
"""

import sys
import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
import yaml
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Suppress CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path="real_model_config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"❌ Error loading config: {e}")
        logger.info("⚠️ Using default configuration...")
        return {
            "models": {
                "base": {"path": "microsoft/DialoGPT-medium"},
                "detoxified": {"path": "microsoft/DialoGPT-medium"}
            },
            "generation": {"max_new_tokens": 50, "temperature": 0.7},
            "dataset": {"name": "allenai/real-toxicity-prompts", "split": "train", "sample_size": 5},
            "classifiers": {
                "toxic_bert": {"model": "unitary/toxic-bert"},
                "roberta_toxicity": {"model": "s-nlp/roberta_toxicity_classifier"},
                "dynabench_hate": {"model": "facebook/roberta-hate-speech-dynabench-r4-target"}
            },
            "output": {"directory": "real_model_results"}
        }


def load_models(config):
    """Load base and detoxified models."""
    logger.info("🔧 Loading language models...")
    
    models = {}
    tokenizers = {}
    
    for model_name, model_config in config["models"].items():
        try:
            model_path = model_config["path"]
            logger.info(f"📥 Loading {model_name} model: {model_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
            tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            
            models[model_name] = model
            tokenizers[model_name] = tokenizer
            logger.info(f"✅ {model_name} model loaded")
            
        except Exception as e:
            logger.error(f"❌ Error loading {model_name} model: {e}")
            logger.info(f"⚠️ Using mock {model_name} model for testing...")
            models[model_name] = None
            tokenizers[model_name] = None
    
    return models, tokenizers


def load_classifiers(config):
    """Load toxicity classifiers."""
    logger.info("🔧 Loading toxicity classifiers...")
    
    classifiers = {}
    
    for classifier_name, classifier_config in config["classifiers"].items():
        try:
            logger.info(f"📥 Loading {classifier_name} classifier...")
            
            # Get classifier parameters
            model_name = classifier_config["model"]
            device = classifier_config.get("device", -1)
            return_all_scores = classifier_config.get("return_all_scores", False)
            
            # Create pipeline
            if return_all_scores:
                classifier = pipeline(
                    "text-classification",
                    model=model_name,
                    return_all_scores=True,
                    device=device
                )
            else:
                classifier = pipeline(
                    "text-classification",
                    model=model_name,
                    device=device
                )
            
            classifiers[classifier_name] = classifier
            logger.info(f"✅ Loaded {classifier_name} classifier (device: {device})")
            
        except Exception as e:
            logger.error(f"❌ Error loading {classifier_name} classifier: {e}")
            logger.info(f"⚠️ Falling back to CPU for {classifier_name}...")
            
            # Fallback to CPU
            try:
                if classifier_config.get("return_all_scores", False):
                    classifier = pipeline(
                        "text-classification",
                        model=model_name,
                        return_all_scores=True,
                        device=-1
                    )
                else:
                    classifier = pipeline(
                        "text-classification",
                        model=model_name,
                        device=-1
                    )
                
                classifiers[classifier_name] = classifier
                logger.info(f"✅ Loaded {classifier_name} classifier on CPU")
                
            except Exception as e2:
                logger.error(f"❌ Failed to load {classifier_name} even on CPU: {e2}")
                classifiers[classifier_name] = None
    
    return classifiers


def generate_outputs(models, tokenizers, prompts, max_new_tokens=50):
    """Generate outputs from models."""
    logger.info("🔄 Generating outputs from models...")
    
    outputs = {}
    
    for model_name, model in models.items():
        if model is None:
            logger.warning(f"⚠️ Model {model_name} not loaded, using mock outputs")
            outputs[model_name] = [f"Mock {model_name} output {i+1}" for i in range(len(prompts))]
            continue
        
        tokenizer = tokenizers[model_name]
        model.eval()
        
        logger.info(f"  Generating with {model_name}...")
        model_outputs = []
        
        for i, prompt in enumerate(prompts):
            try:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    generated_ids = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Extract only the generated tokens
                gen_only = generated_ids[:, inputs["input_ids"].shape[1]:]
                output = tokenizer.decode(gen_only[0], skip_special_tokens=True)
                model_outputs.append(output)
                
                # Progress tracking
                if (i + 1) % 5 == 0:
                    logger.info(f"    Generated {i + 1}/{len(prompts)} outputs for {model_name}")
                
                # Clear GPU memory periodically
                if torch.cuda.is_available() and (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.warning(f"⚠️ Error generating output {i} for {model_name}: {e}")
                model_outputs.append(f"Error generating output {i}")
        
        outputs[model_name] = model_outputs
        logger.info(f"✅ Generated {len(model_outputs)} outputs for {model_name}")
        
        # Clear GPU memory after each model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return outputs


def classify_texts(classifiers, texts, text_type):
    """Classify texts and return results as dictionaries."""
    logger.info(f"🔍 Classifying {text_type}...")
    
    results = {}
    
    for classifier_name, classifier in classifiers.items():
        if classifier is None:
            logger.warning(f"⚠️ Classifier {classifier_name} not loaded, skipping")
            continue
        
        logger.info(f"  Running {classifier_name}...")
        classifier_results = []
        
        # Log first few results for debugging
        debug_count = 0
        
        for text in texts:
            try:
                if classifier_name == "toxic_bert":
                    # Toxic-bert returns all scores for all categories
                    predictions = classifier(text, truncation=True, max_length=512)
                    if isinstance(predictions, list) and len(predictions) > 0:
                        # Convert to dictionary format
                        result = {}
                        for pred in predictions[0]:  # Take first (and only) prediction
                            label = pred["label"].lower()
                            score = pred["score"]
                            result[label] = score
                        classifier_results.append(result)
                    else:
                        classifier_results.append({})
                else:
                    # Single-label classifiers - get all scores
                    predictions = classifier(text, truncation=True, max_length=512, return_all_scores=True)
                    if isinstance(predictions, list) and len(predictions) > 0:
                        # Convert to dictionary format with all scores
                        result = {}
                        for pred in predictions[0]:  # Take first (and only) prediction
                            label = pred["label"].lower()
                            score = pred["score"]
                            
                            # Map labels to consistent names
                            if classifier_name == "roberta_toxicity":
                                if "toxic" in label:
                                    label = "toxic"
                                elif "neutral" in label:
                                    label = "neutral"
                            elif classifier_name == "dynabench_hate":
                                if "hate" in label and "not" not in label:
                                    label = "hate"
                                elif "not" in label or "non" in label:
                                    label = "not_hate"
                            
                            result[label] = score
                        classifier_results.append(result)
                        
                        # Debug logging for first few results
                        if debug_count < 3:
                            logger.info(f"    Sample {debug_count + 1} {classifier_name} results: {result}")
                            debug_count += 1
                    else:
                        classifier_results.append({})
                        
            except Exception as e:
                logger.warning(f"⚠️ Error classifying text with {classifier_name}: {e}")
                classifier_results.append({})
        
        results[classifier_name] = classifier_results
    
    return results


def create_comprehensive_results(prompts, model_outputs, classifiers):
    """Create comprehensive results with all classifications."""
    logger.info("📊 Creating comprehensive results...")
    
    # Classify prompts
    prompt_classifications = classify_texts(classifiers, prompts, "prompts")
    
    # Classify outputs for each model
    output_classifications = {}
    full_text_classifications = {}
    
    for model_name, outputs in model_outputs.items():
        # Classify outputs
        output_classifications[model_name] = classify_texts(classifiers, outputs, f"outputs_{model_name}")
        
        # Create full texts (prompt + output)
        full_texts = [f"{prompt} {output}" for prompt, output in zip(prompts, outputs)]
        full_text_classifications[model_name] = classify_texts(classifiers, full_texts, f"full_texts_{model_name}")
    
    # Create separate DataFrames for each model with generic column names
    model_dfs = {}
    
    for model_name, outputs in model_outputs.items():
        df_data = []
        for i in range(len(prompts)):
            row = {
                "prompt": prompts[i],
                "prompt_index": i,
                "output": outputs[i],
                "full_text": f"{prompts[i]} {outputs[i]}"
            }
            
            # Add prompt classifications
            for classifier_name, results in prompt_classifications.items():
                if results and i < len(results):
                    row[f"prompt_{classifier_name}_results"] = results[i]
            
            # Add output classifications for this model
            if model_name in output_classifications:
                for classifier_name, results in output_classifications[model_name].items():
                    if results and i < len(results):
                        row[f"output_{classifier_name}_results"] = results[i]
            
            # Add full text classifications for this model
            if model_name in full_text_classifications:
                for classifier_name, results in full_text_classifications[model_name].items():
                    if results and i < len(results):
                        row[f"full_text_{classifier_name}_results"] = results[i]
            
            df_data.append(row)
        
        model_dfs[model_name] = pd.DataFrame(df_data)
        logger.info(f"✅ Created DataFrame for {model_name} with {len(df_data)} rows and {len(model_dfs[model_name].columns)} columns")
    
    return model_dfs


def save_results(model_dfs, model_outputs, output_dir="real_model_results"):
    """Save results to separate files."""
    logger.info(f"💾 Saving results to {output_dir}...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save separate CSV files for each model with model names
    for model_name, model_df in model_dfs.items():
        # Convert dictionaries to strings for CSV
        model_df_csv = model_df.copy()
        for col in model_df_csv.columns:
            if col.endswith('_results'):
                model_df_csv[col] = model_df_csv[col].apply(lambda x: str(x) if isinstance(x, dict) else x)
        
        # Save model-specific CSV with model name
        model_csv_path = output_path / f"{model_name}_results.csv"
        model_df_csv.to_csv(model_csv_path, index=False)
        logger.info(f"✅ Saved {model_name} results to {model_csv_path}")
        
        # Save model-specific JSON with model name
        model_json_path = output_path / f"{model_name}_results.json"
        model_df.to_json(model_json_path, orient='records', indent=2)
        logger.info(f"✅ Saved {model_name} results to {model_json_path}")
    
    # Create comprehensive DataFrame for comparison (optional)
    comprehensive_data = []
    for model_name, model_df in model_dfs.items():
        for _, row in model_df.iterrows():
            comprehensive_row = {
                "model": model_name,
                "prompt": row["prompt"],
                "prompt_index": row["prompt_index"],
                "output": row["output"],
                "full_text": row["full_text"]
            }
            # Add all classification results
            for col in model_df.columns:
                if col.endswith('_results'):
                    comprehensive_row[col] = row[col]
            comprehensive_data.append(comprehensive_row)
    
    comprehensive_df = pd.DataFrame(comprehensive_data)
    comprehensive_path = output_path / "comprehensive_results.csv"
    comprehensive_csv = comprehensive_df.copy()
    for col in comprehensive_csv.columns:
        if col.endswith('_results'):
            comprehensive_csv[col] = comprehensive_csv[col].apply(lambda x: str(x) if isinstance(x, dict) else x)
    comprehensive_csv.to_csv(comprehensive_path, index=False)
    logger.info(f"✅ Saved comprehensive results to {comprehensive_path}")
    
    # Save comprehensive results as JSON
    json_path = output_path / "comprehensive_results.json"
    comprehensive_df.to_json(json_path, orient='records', indent=2)
    logger.info(f"✅ Saved comprehensive results to {json_path}")
    
    # Save model outputs separately with model names
    for model_name, outputs in model_outputs.items():
        model_output_path = output_path / f"{model_name}_outputs.txt"
        with open(model_output_path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {model_name}\n")
            f.write("=" * 50 + "\n\n")
            for i, output in enumerate(outputs):
                f.write(f"Output {i+1}:\n{output}\n\n")
        logger.info(f"✅ Saved {model_name} outputs to {model_output_path}")
    
    # Save prompts
    prompts_path = output_path / "prompts.txt"
    # Get prompts from the first model DataFrame
    first_model_df = list(model_dfs.values())[0]
    with open(prompts_path, 'w', encoding='utf-8') as f:
        for i, prompt in enumerate(first_model_df['prompt']):
            f.write(f"Prompt {i+1}:\n{prompt}\n\n")
    logger.info(f"✅ Saved prompts to {prompts_path}")
    
    # Save model mapping for reference
    model_mapping = {}
    for model_name in model_outputs.keys():
        model_mapping[model_name] = {
            "name": model_name,
            "description": f"Model: {model_name}"
        }
    
    mapping_path = output_path / "model_mapping.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(model_mapping, f, indent=2)
    logger.info(f"✅ Saved model mapping to {mapping_path}")
    
    # Save classification summary
    summary_data = {}
    for model_name, model_df in model_dfs.items():
        summary_data[model_name] = {}
        
        for col in model_df.columns:
            if col.endswith('_results'):
                # Extract classifier and text type from column name
                parts = col.split('_')
                if len(parts) >= 3:
                    text_type = parts[0]  # prompt, output, or full_text
                    classifier = parts[1]  # classifier name
                    
                    if classifier not in summary_data[model_name]:
                        summary_data[model_name][classifier] = {}
                    
                    if text_type not in summary_data[model_name][classifier]:
                        summary_data[model_name][classifier][text_type] = {}
                    
                    # Calculate average scores for each category
                    valid_results = [r for r in model_df[col] if isinstance(r, dict) and r]
                    if valid_results:
                        all_categories = set()
                        for result in valid_results:
                            all_categories.update(result.keys())
                        
                        for category in all_categories:
                            scores = [result.get(category, 0.0) for result in valid_results]
                            summary_data[model_name][classifier][text_type][category] = {
                                "mean": np.mean(scores),
                                "std": np.std(scores),
                                "min": np.min(scores),
                                "max": np.max(scores)
                            }
    
    summary_path = output_path / "classification_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"✅ Saved classification summary to {summary_path}")
    
    # Create model comparison summary
    comparison_data = {}
    models_list = list(model_outputs.keys())
    
    if len(models_list) >= 2:
        base_model = models_list[0]
        detox_model = models_list[1]
        
        comparison_data["model_comparison"] = {
            "base_model": base_model,
            "detoxified_model": detox_model,
            "improvements": {}
        }
        
        # Calculate improvements for each classifier and text type
        for classifier_name in summary_data.keys():
            if classifier_name in summary_data:
                classifier_data = summary_data[classifier_name]
                
                for text_type in ["output", "full_text"]:
                    if text_type in classifier_data:
                        text_data = classifier_data[text_type]
                        
                        for category, stats in text_data.items():
                            base_key = f"{base_model}_{category}"
                            detox_key = f"{detox_model}_{category}"
                            
                            if base_key in stats and detox_key in stats:
                                improvement = stats[base_key]["mean"] - stats[detox_key]["mean"]
                                comparison_data["model_comparison"]["improvements"][f"{classifier_name}_{text_type}_{category}"] = {
                                    "improvement": improvement,
                                    "base_mean": stats[base_key]["mean"],
                                    "detoxified_mean": stats[detox_key]["mean"],
                                    "percentage_improvement": (improvement / stats[base_key]["mean"]) * 100 if stats[base_key]["mean"] > 0 else 0
                                }
        
        comparison_path = output_path / "model_comparison.json"
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2)
        logger.info(f"✅ Saved model comparison to {comparison_path}")
    
    return output_path


def print_classifier_summary(df):
    """Print summary of classifier scores for debugging."""
    logger.info("\n🔍 CLASSIFIER SCORE SUMMARY:")
    logger.info("=" * 40)
    
    for col in df.columns:
        if col.endswith('_results'):
            logger.info(f"\n📊 {col}:")
            valid_results = [r for r in df[col] if isinstance(r, dict) and r]
            if valid_results:
                # Get all categories
                all_categories = set()
                for result in valid_results:
                    all_categories.update(result.keys())
                
                for category in sorted(all_categories):
                    scores = [result.get(category, 0.0) for result in valid_results]
                    mean_score = np.mean(scores)
                    logger.info(f"  {category}: {mean_score:.4f} (mean)")
            else:
                logger.info("  No valid results")


def create_toxicity_plots(model_dfs, output_path):
    """Create scatter plots showing toxicity reduction across epochs."""
    logger.info("📊 Creating toxicity reduction plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Extract epoch numbers and create mapping
    model_epochs = {}
    for model_name in model_dfs.keys():
        if model_name == "base":
            model_epochs[model_name] = 0  # Base model = epoch 0
        elif "epoch" in model_name:
            # Extract epoch number from name like "detox_epoch_20"
            epoch_num = int(model_name.split("_")[-1])
            model_epochs[model_name] = epoch_num
        else:
            # For any other models, assign a default epoch
            model_epochs[model_name] = 100
    
    # Sort models by epoch
    sorted_models = sorted(model_epochs.items(), key=lambda x: x[1])
    model_names = [name for name, _ in sorted_models]
    epochs = [epoch for _, epoch in sorted_models]
    
    # Create plots for each classifier and text type
    classifiers = ["toxic_bert", "roberta_toxicity", "dynabench_hate"]
    text_types = ["output", "full_text"]
    
    for classifier in classifiers:
        for text_type in text_types:
            logger.info(f"  Creating plot for {classifier} - {text_type}")
            
            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'Toxicity Reduction: {classifier} - {text_type}', fontsize=16, fontweight='bold')
            
            # Plot 1: Mean toxicity across epochs
            mean_scores = []
            for model_name in model_names:
                if model_name in model_dfs:
                    df = model_dfs[model_name]
                    col_name = f"{text_type}_{classifier}_results"
                    
                    if col_name in df.columns:
                        # Extract toxicity scores
                        scores = []
                        for _, row in df.iterrows():
                            if isinstance(row[col_name], dict):
                                if classifier == "toxic_bert":
                                    # Use 'toxic' category for toxic-bert
                                    scores.append(row[col_name].get('toxic', 0.0))
                                elif classifier == "roberta_toxicity":
                                    # Use 'toxic' category for roberta
                                    scores.append(row[col_name].get('toxic', 0.0))
                                elif classifier == "dynabench_hate":
                                    # Use 'hate' category for dynabench
                                    scores.append(row[col_name].get('hate', 0.0))
                        
                        if scores:
                            mean_scores.append(np.mean(scores))
                        else:
                            mean_scores.append(0.0)
                    else:
                        mean_scores.append(0.0)
                else:
                    mean_scores.append(0.0)
            
            # Plot mean toxicity progression
            axes[0].plot(epochs, mean_scores, 'o-', linewidth=2, markersize=8, label='Mean Toxicity')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Mean Toxicity Score')
            axes[0].set_title(f'Mean Toxicity Across Epochs')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            
            # Plot 2: Scatter plot of individual prompt toxicity
            all_scores = []
            all_epochs = []
            all_prompts = []
            
            for i, model_name in enumerate(model_names):
                if model_name in model_dfs:
                    df = model_dfs[model_name]
                    col_name = f"{text_type}_{classifier}_results"
                    
                    if col_name in df.columns:
                        for j, (_, row) in enumerate(df.iterrows()):
                            if isinstance(row[col_name], dict):
                                if classifier == "toxic_bert":
                                    score = row[col_name].get('toxic', 0.0)
                                elif classifier == "roberta_toxicity":
                                    score = row[col_name].get('toxic', 0.0)
                                elif classifier == "dynabench_hate":
                                    score = row[col_name].get('hate', 0.0)
                                else:
                                    score = 0.0
                                
                                all_scores.append(score)
                                all_epochs.append(epochs[i])
                                all_prompts.append(j)
            
            # Create scatter plot
            scatter = axes[1].scatter(all_epochs, all_scores, c=all_prompts, cmap='viridis', alpha=0.6, s=30)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Toxicity Score')
            axes[1].set_title(f'Individual Prompt Toxicity')
            axes[1].grid(True, alpha=0.3)
            
            # Add colorbar for prompt indices
            cbar = plt.colorbar(scatter, ax=axes[1])
            cbar.set_label('Prompt Index')
            
            # Add improvement annotations
            if len(mean_scores) > 1:
                improvement = mean_scores[0] - mean_scores[-1]  # Base - Final
                improvement_pct = (improvement / mean_scores[0]) * 100 if mean_scores[0] > 0 else 0
                axes[0].text(0.02, 0.98, f'Total Reduction: {improvement:.3f} ({improvement_pct:.1f}%)', 
                           transform=axes[0].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot
            plot_path = output_path / f"toxicity_reduction_{classifier}_{text_type}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"    Saved plot to {plot_path}")
            plt.close()
    
    # Create a summary plot showing all classifiers for outputs
    logger.info("  Creating summary plot for outputs")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Toxicity Reduction Summary - Outputs', fontsize=16, fontweight='bold')
    
    for i, classifier in enumerate(classifiers):
        mean_scores = []
        for model_name in model_names:
            if model_name in model_dfs:
                df = model_dfs[model_name]
                col_name = f"output_{classifier}_results"
                
                if col_name in df.columns:
                    scores = []
                    for _, row in df.iterrows():
                        if isinstance(row[col_name], dict):
                            if classifier == "toxic_bert":
                                scores.append(row[col_name].get('toxic', 0.0))
                            elif classifier == "roberta_toxicity":
                                scores.append(row[col_name].get('toxic', 0.0))
                            elif classifier == "dynabench_hate":
                                scores.append(row[col_name].get('hate', 0.0))
                    
                    if scores:
                        mean_scores.append(np.mean(scores))
                    else:
                        mean_scores.append(0.0)
                else:
                    mean_scores.append(0.0)
            else:
                mean_scores.append(0.0)
        
        axes[i].plot(epochs, mean_scores, 'o-', linewidth=2, markersize=8)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Mean Toxicity Score')
        axes[i].set_title(f'{classifier.replace("_", " ").title()}')
        axes[i].grid(True, alpha=0.3)
        
        # Add improvement annotation
        if len(mean_scores) > 1:
            improvement = mean_scores[0] - mean_scores[-1]
            improvement_pct = (improvement / mean_scores[0]) * 100 if mean_scores[0] > 0 else 0
            axes[i].text(0.02, 0.98, f'Reduction: {improvement_pct:.1f}%', 
                       transform=axes[i].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    summary_plot_path = output_path / "toxicity_reduction_summary_outputs.png"
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"    Saved summary plot to {summary_plot_path}")
    plt.close()
    
    logger.info("✅ All toxicity reduction plots created!")


def create_prompt_comparison_plots(model_dfs, output_path):
    """Create plots showing toxicity scores for each prompt across all models."""
    logger.info("📊 Creating prompt comparison plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Extract epoch numbers and create mapping
    model_epochs = {}
    for model_name in model_dfs.keys():
        if model_name == "base":
            model_epochs[model_name] = 0  # Base model = epoch 0
        elif "epoch" in model_name:
            # Extract epoch number from name like "detox_epoch_20"
            epoch_num = int(model_name.split("_")[-1])
            model_epochs[model_name] = epoch_num
        else:
            # For any other models, assign a default epoch
            model_epochs[model_name] = 100
    
    # Sort models by epoch
    sorted_models = sorted(model_epochs.items(), key=lambda x: x[1])
    model_names = [name for name, _ in sorted_models]
    epochs = [epoch for _, epoch in sorted_models]
    
    # Get the first model to determine number of prompts
    first_model_name = list(model_dfs.keys())[0]
    first_df = model_dfs[first_model_name]
    num_prompts = len(first_df)
    
    # Create plots for each classifier and text type
    classifiers = ["toxic_bert", "roberta_toxicity", "dynabench_hate"]
    text_types = ["output", "full_text"]
    
    for classifier in classifiers:
        for text_type in text_types:
            logger.info(f"  Creating prompt comparison for {classifier} - {text_type}")
            
            # Create a large figure with subplots for each prompt
            num_cols = 5  # 5 columns
            num_rows = (num_prompts + num_cols - 1) // num_cols  # Calculate rows needed
            
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows))
            fig.suptitle(f'Prompt-by-Prompt Toxicity Comparison: {classifier} - {text_type}', 
                        fontsize=16, fontweight='bold')
            
            # Flatten axes for easier indexing
            if num_rows == 1:
                axes = [axes] if num_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            # Plot each prompt
            for prompt_idx in range(num_prompts):
                ax = axes[prompt_idx]
                
                # Collect toxicity scores for this prompt across all models
                prompt_scores = []
                prompt_epochs = []
                
                for model_name in model_names:
                    if model_name in model_dfs:
                        df = model_dfs[model_name]
                        col_name = f"{text_type}_{classifier}_results"
                        
                        if col_name in df.columns and prompt_idx < len(df):
                            row = df.iloc[prompt_idx]
                            if isinstance(row[col_name], dict):
                                if classifier == "toxic_bert":
                                    score = row[col_name].get('toxic', 0.0)
                                elif classifier == "roberta_toxicity":
                                    score = row[col_name].get('toxic', 0.0)
                                elif classifier == "dynabench_hate":
                                    score = row[col_name].get('hate', 0.0)
                                else:
                                    score = 0.0
                                
                                prompt_scores.append(score)
                                prompt_epochs.append(model_epochs[model_name])
                
                # Plot this prompt's progression
                if prompt_scores:
                    ax.plot(prompt_epochs, prompt_scores, 'o-', linewidth=2, markersize=6)
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Toxicity Score')
                    ax.set_title(f'Prompt {prompt_idx + 1}')
                    ax.grid(True, alpha=0.3)
                    
                    # Add improvement annotation
                    if len(prompt_scores) > 1:
                        improvement = prompt_scores[0] - prompt_scores[-1]  # Base - Final
                        improvement_pct = (improvement / prompt_scores[0]) * 100 if prompt_scores[0] > 0 else 0
                        ax.text(0.02, 0.98, f'{improvement_pct:.0f}%', 
                               transform=ax.transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightgreen' if improvement > 0 else 'lightcoral', alpha=0.8))
                
                # Hide unused subplots
                if prompt_idx >= num_prompts:
                    ax.set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = output_path / f"prompt_comparison_{classifier}_{text_type}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"    Saved prompt comparison plot to {plot_path}")
            plt.close()
    
    # Create a summary heatmap showing improvement percentages
    logger.info("  Creating improvement heatmap")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Toxicity Reduction Heatmap by Prompt', fontsize=16, fontweight='bold')
    
    for i, classifier in enumerate(classifiers):
        for j, text_type in enumerate(text_types):
            ax = axes[j, i]
            
            # Calculate improvement matrix
            improvement_matrix = []
            for prompt_idx in range(min(20, num_prompts)):  # Show first 20 prompts
                prompt_improvements = []
                
                for model_name in model_names:
                    if model_name in model_dfs:
                        df = model_dfs[model_name]
                        col_name = f"{text_type}_{classifier}_results"
                        
                        if col_name in df.columns and prompt_idx < len(df):
                            row = df.iloc[prompt_idx]
                            if isinstance(row[col_name], dict):
                                if classifier == "toxic_bert":
                                    score = row[col_name].get('toxic', 0.0)
                                elif classifier == "roberta_toxicity":
                                    score = row[col_name].get('toxic', 0.0)
                                elif classifier == "dynabench_hate":
                                    score = row[col_name].get('hate', 0.0)
                                else:
                                    score = 0.0
                                
                                prompt_improvements.append(score)
                            else:
                                prompt_improvements.append(0.0)
                        else:
                            prompt_improvements.append(0.0)
                    else:
                        prompt_improvements.append(0.0)
                
                improvement_matrix.append(prompt_improvements)
            
            # Create heatmap
            if improvement_matrix:
                im = ax.imshow(improvement_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Prompt Index')
                ax.set_title(f'{classifier.replace("_", " ").title()} - {text_type}')
                
                # Set x-axis labels
                ax.set_xticks(range(len(epochs)))
                ax.set_xticklabels(epochs)
                
                # Set y-axis labels
                ax.set_yticks(range(min(20, num_prompts)))
                ax.set_yticklabels(range(1, min(21, num_prompts + 1)))
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Toxicity Score')
    
    plt.tight_layout()
    heatmap_path = output_path / "toxicity_improvement_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    logger.info(f"    Saved improvement heatmap to {heatmap_path}")
    plt.close()
    
    logger.info("✅ All prompt comparison plots created!")


def create_interactive_plots(model_dfs, output_path):
    """Create interactive plots showing toxicity improvements and distributions."""
    logger.info("📊 Creating interactive plots...")
    
    # Extract epoch numbers and create mapping
    model_epochs = {}
    for model_name in model_dfs.keys():
        if model_name == "base":
            model_epochs[model_name] = 0  # Base model = epoch 0
        elif "epoch" in model_name:
            # Extract epoch number from name like "detox_epoch_20"
            epoch_num = int(model_name.split("_")[-1])
            model_epochs[model_name] = epoch_num
        else:
            # For any other models, assign a default epoch
            model_epochs[model_name] = 100
    
    # Sort models by epoch
    sorted_models = sorted(model_epochs.items(), key=lambda x: x[1])
    model_names = [name for name, _ in sorted_models]
    epochs = [epoch for _, epoch in sorted_models]
    
    # Get detoxified models (exclude base)
    detox_models = [name for name in model_names if name != "base"]
    detox_epochs = [epoch for name, epoch in sorted_models if name != "base"]
    
    # Create plots for each classifier and text type
    classifiers = ["toxic_bert", "roberta_toxicity", "dynabench_hate"]
    text_types = ["output", "full_text"]
    
    for classifier in classifiers:
        for text_type in text_types:
            logger.info(f"  Creating interactive plots for {classifier} - {text_type}")
            
            # 1. Distribution of improvements (base - detoxified)
            fig_dist = make_subplots(
                rows=1, cols=1,
                subplot_titles=[f'Distribution of Toxicity Improvements: {classifier} - {text_type}']
            )
            
            # Get base model scores
            base_df = model_dfs["base"]
            base_col = f"{text_type}_{classifier}_results"
            base_scores = []
            
            if base_col in base_df.columns:
                for _, row in base_df.iterrows():
                    if isinstance(row[base_col], dict):
                        if classifier == "toxic_bert":
                            score = row[base_col].get('toxic', 0.0)
                        elif classifier == "roberta_toxicity":
                            score = row[base_col].get('toxic', 0.0)
                        elif classifier == "dynabench_hate":
                            score = row[base_col].get('hate', 0.0)
                        else:
                            score = 0.0
                        base_scores.append(score)
            
            # Calculate improvements for each detoxified model
            for detox_model in detox_models:
                detox_df = model_dfs[detox_model]
                detox_col = f"{text_type}_{classifier}_results"
                improvements = []
                
                if detox_col in detox_df.columns and base_col in base_df.columns:
                    for i, (_, row) in enumerate(detox_df.iterrows()):
                        if i < len(base_scores) and isinstance(row[detox_col], dict):
                            if classifier == "toxic_bert":
                                detox_score = row[detox_col].get('toxic', 0.0)
                            elif classifier == "roberta_toxicity":
                                detox_score = row[detox_col].get('toxic', 0.0)
                            elif classifier == "dynabench_hate":
                                detox_score = row[detox_col].get('hate', 0.0)
                            else:
                                detox_score = 0.0
                            
                            improvement = base_scores[i] - detox_score
                            improvements.append(improvement)
                
                if improvements:
                    fig_dist.add_trace(
                        go.Histogram(
                            x=improvements,
                            name=f'Epoch {model_epochs[detox_model]}',
                            opacity=0.7,
                            nbinsx=20
                        )
                    )
            
            fig_dist.update_layout(
                title=f'Distribution of Toxicity Improvements: {classifier} - {text_type}',
                xaxis_title='Improvement (Base - Detoxified)',
                yaxis_title='Count',
                barmode='overlay',
                showlegend=True
            )
            
            # Save distribution plot
            dist_path = output_path / f"interactive_distribution_{classifier}_{text_type}.html"
            fig_dist.write_html(str(dist_path))
            logger.info(f"    Saved distribution plot to {dist_path}")
            
            # 2. Scatter plot: Base vs Detoxified (interactive selection)
            fig_scatter = make_subplots(
                rows=1, cols=1,
                subplot_titles=[f'Base vs Detoxified Toxicity: {classifier} - {text_type}']
            )
            
            # Add base scores on x-axis
            fig_scatter.add_trace(
                go.Scatter(
                    x=base_scores,
                    y=base_scores,
                    mode='markers',
                    name='Base (reference line)',
                    line=dict(color='gray', dash='dash'),
                    showlegend=True
                )
            )
            
            # Add detoxified models
            for detox_model in detox_models:
                detox_df = model_dfs[detox_model]
                detox_col = f"{text_type}_{classifier}_results"
                detox_scores = []
                
                if detox_col in detox_df.columns:
                    for i, (_, row) in enumerate(detox_df.iterrows()):
                        if i < len(base_scores) and isinstance(row[detox_col], dict):
                            if classifier == "toxic_bert":
                                score = row[detox_col].get('toxic', 0.0)
                            elif classifier == "roberta_toxicity":
                                score = row[detox_col].get('toxic', 0.0)
                            elif classifier == "dynabench_hate":
                                score = row[detox_col].get('hate', 0.0)
                            else:
                                score = 0.0
                            detox_scores.append(score)
                
                if detox_scores:
                    fig_scatter.add_trace(
                        go.Scatter(
                            x=base_scores,
                            y=detox_scores,
                            mode='markers',
                            name=f'Epoch {model_epochs[detox_model]}',
                            marker=dict(size=8, opacity=0.7),
                            hovertemplate='<b>Prompt %{text}</b><br>Base: %{x:.3f}<br>Detoxified: %{y:.3f}<extra></extra>',
                            text=[f"{i+1}" for i in range(len(base_scores))]
                        )
                    )
            
            fig_scatter.update_layout(
                title=f'Base vs Detoxified Toxicity: {classifier} - {text_type}',
                xaxis_title='Base Toxicity Score',
                yaxis_title='Detoxified Toxicity Score',
                showlegend=True
            )
            
            # Add diagonal line for reference
            max_score = max(max(base_scores) if base_scores else 0, 1.0)
            fig_scatter.add_trace(
                go.Scatter(
                    x=[0, max_score],
                    y=[0, max_score],
                    mode='lines',
                    name='No Change',
                    line=dict(color='red', dash='dash'),
                    showlegend=True
                )
            )
            
            # Save scatter plot
            scatter_path = output_path / f"interactive_scatter_{classifier}_{text_type}.html"
            fig_scatter.write_html(str(scatter_path))
            logger.info(f"    Saved scatter plot to {scatter_path}")
            
            # 3. Tracking plot: Toxicity over epochs for each prompt
            fig_tracking = make_subplots(
                rows=1, cols=1,
                subplot_titles=[f'Toxicity Tracking Over Epochs: {classifier} - {text_type}']
            )
            
            # Plot each prompt's progression
            for prompt_idx in range(min(20, len(base_scores))):  # Show first 20 prompts
                prompt_scores = []
                prompt_epochs = []
                
                for model_name in model_names:
                    if model_name in model_dfs:
                        df = model_dfs[model_name]
                        col_name = f"{text_type}_{classifier}_results"
                        
                        if col_name in df.columns and prompt_idx < len(df):
                            row = df.iloc[prompt_idx]
                            if isinstance(row[col_name], dict):
                                if classifier == "toxic_bert":
                                    score = row[col_name].get('toxic', 0.0)
                                elif classifier == "roberta_toxicity":
                                    score = row[col_name].get('toxic', 0.0)
                                elif classifier == "dynabench_hate":
                                    score = row[col_name].get('hate', 0.0)
                                else:
                                    score = 0.0
                                
                                prompt_scores.append(score)
                                prompt_epochs.append(model_epochs[model_name])
                
                if prompt_scores:
                    fig_tracking.add_trace(
                        go.Scatter(
                            x=prompt_epochs,
                            y=prompt_scores,
                            mode='lines+markers',
                            name=f'Prompt {prompt_idx + 1}',
                            line=dict(width=2),
                            marker=dict(size=6),
                            hovertemplate='<b>Prompt %{fullData.name}</b><br>Epoch: %{x}<br>Toxicity: %{y:.3f}<extra></extra>'
                        )
                    )
            
            fig_tracking.update_layout(
                title=f'Toxicity Tracking Over Epochs: {classifier} - {text_type}',
                xaxis_title='Epoch',
                yaxis_title='Toxicity Score',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            # Save tracking plot
            tracking_path = output_path / f"interactive_tracking_{classifier}_{text_type}.html"
            fig_tracking.write_html(str(tracking_path))
            logger.info(f"    Saved tracking plot to {tracking_path}")
    
    # Create a comprehensive dashboard
    logger.info("  Creating comprehensive dashboard")
    fig_dashboard = make_subplots(
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
                if model_name in model_dfs:
                    df = model_dfs[model_name]
                    col_name = f"{text_type}_{classifier}_results"
                    
                    if col_name in df.columns:
                        scores = []
                        for _, row_data in df.iterrows():
                            if isinstance(row_data[col_name], dict):
                                if classifier == "toxic_bert":
                                    scores.append(row_data[col_name].get('toxic', 0.0))
                                elif classifier == "roberta_toxicity":
                                    scores.append(row_data[col_name].get('toxic', 0.0))
                                elif classifier == "dynabench_hate":
                                    scores.append(row_data[col_name].get('hate', 0.0))
                        
                        if scores:
                            mean_scores.append(np.mean(scores))
                        else:
                            mean_scores.append(0.0)
                    else:
                        mean_scores.append(0.0)
                else:
                    mean_scores.append(0.0)
            
            if mean_scores:
                fig_dashboard.add_trace(
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
                    fig_dashboard.add_annotation(
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
    
    fig_dashboard.update_layout(
        title='Comprehensive Toxicity Reduction Dashboard',
        height=800,
        showlegend=False
    )
    
    # Update axes labels
    for i in range(1, 3):
        for j in range(1, 4):
            fig_dashboard.update_xaxes(title_text="Epoch", row=i, col=j)
            fig_dashboard.update_yaxes(title_text="Mean Toxicity", row=i, col=j)
    
    # Save dashboard
    dashboard_path = output_path / f"comprehensive_dashboard.html"
    fig_dashboard.write_html(str(dashboard_path))
    logger.info(f"    Saved comprehensive dashboard to {dashboard_path}")
    
    logger.info("✅ All interactive plots created!")


def main():
    """Run real end-to-end test with actual models and classifiers."""
    logger.info("🚀 Starting Real End-to-End Model Test")
    logger.info("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"🔥 GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"🔥 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.info("💻 Using CPU for computation")
    
    try:
        # Load configuration
        config = load_config()
        
        # Load AllenAI dataset
        logger.info("📥 Loading AllenAI dataset...")
        dataset = load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])
        
        # Filter for toxic prompts if specified
        if config["dataset"].get("filter_toxic", False):
            logger.info("🔍 Filtering for toxic prompts...")
            min_toxicity = config["dataset"].get("min_toxicity_score", 0.5)
            
            # Filter dataset by toxicity score
            toxic_prompts = []
            for item in dataset:
                toxicity_score = item.get("prompt", {}).get("toxicity")
                # Skip items with no toxicity score or None values
                if toxicity_score is not None and toxicity_score >= min_toxicity:
                    toxic_prompts.append(item)
            
            logger.info(f"📊 Found {len(toxic_prompts)} prompts with toxicity >= {min_toxicity}")
            
            # Take sample from toxic prompts
            sample_size = min(config["dataset"]["sample_size"], len(toxic_prompts))
            sample_data = toxic_prompts[:sample_size]
            prompts = [item["prompt"]["text"] for item in sample_data]
            
            logger.info(f"✅ Loaded {len(prompts)} toxic prompts (toxicity >= {min_toxicity})")
            
            # Show toxicity statistics
            toxicity_scores = [item["prompt"]["toxicity"] for item in sample_data if item["prompt"]["toxicity"] is not None]
            if toxicity_scores:
                logger.info(f"📊 Toxicity statistics:")
                logger.info(f"  - Mean toxicity: {np.mean(toxicity_scores):.3f}")
                logger.info(f"  - Min toxicity: {np.min(toxicity_scores):.3f}")
                logger.info(f"  - Max toxicity: {np.max(toxicity_scores):.3f}")
            else:
                logger.info("📊 No toxicity scores available for selected prompts")
        else:
            # Take random sample
            sample_size = config["dataset"]["sample_size"]
            sample_data = dataset.select(range(sample_size))
            prompts = [item["prompt"]["text"] for item in sample_data]
            
            logger.info(f"✅ Loaded {len(prompts)} random prompts from AllenAI dataset")
            
            # Show toxicity statistics for random sample too
            toxicity_scores = [item["prompt"]["toxicity"] for item in sample_data if item["prompt"]["toxicity"] is not None]
            if toxicity_scores:
                logger.info(f"📊 Toxicity statistics:")
                logger.info(f"  - Mean toxicity: {np.mean(toxicity_scores):.3f}")
                logger.info(f"  - Min toxicity: {np.min(toxicity_scores):.3f}")
                logger.info(f"  - Max toxicity: {np.max(toxicity_scores):.3f}")
            else:
                logger.info("📊 No toxicity scores available for selected prompts")
        
        # Load models and classifiers
        models, tokenizers = load_models(config)
        classifiers = load_classifiers(config)
        
        # Generate outputs
        model_outputs = generate_outputs(models, tokenizers, prompts, config["generation"]["max_new_tokens"])
        
        # Create comprehensive results
        model_dfs = create_comprehensive_results(prompts, model_outputs, classifiers)
        
        # Save results
        output_path = save_results(model_dfs, model_outputs, config["output"]["directory"])
        
        # Create toxicity reduction plots
        create_toxicity_plots(model_dfs, output_path)
        
        # Create prompt comparison plots
        create_prompt_comparison_plots(model_dfs, output_path)
        
        # Create interactive plots
        create_interactive_plots(model_dfs, output_path)
        
        # Print classifier summary for first model
        first_model_df = list(model_dfs.values())[0]
        print_classifier_summary(first_model_df)
        
        # Display summary
        logger.info("\n📋 RESULTS SUMMARY:")
        logger.info("=" * 30)
        
        for model_name, outputs in model_outputs.items():
            model_num = list(model_outputs.keys()).index(model_name) + 1
            logger.info(f"\nMODEL {model_num} ({model_name.upper()}):")
            logger.info(f"  Generated {len(outputs)} outputs")
            if outputs:
                logger.info(f"  Sample output: {outputs[0][:100]}...")
        
        # Show classification results for first sample
        first_model_df = list(model_dfs.values())[0]
        first_row = first_model_df.iloc[0]
        logger.info(f"\nFIRST SAMPLE CLASSIFICATIONS:")
        logger.info(f"  Prompt: {first_row['prompt'][:50]}...")
        
        # Show toxic-bert results specifically
        logger.info(f"\nTOXIC-BERT CLASSIFICATIONS (First Sample):")
        for col in first_model_df.columns:
            if col.endswith('_results') and isinstance(first_row[col], dict) and 'toxic_bert' in col:
                logger.info(f"  {col}: {first_row[col]}")
        
        # Show other classifier results
        logger.info(f"\nOTHER CLASSIFIER RESULTS (First Sample):")
        for col in first_model_df.columns:
            if col.endswith('_results') and isinstance(first_row[col], dict) and 'toxic_bert' not in col:
                logger.info(f"  {col}: {first_row[col]}")
        
        # Show available columns
        logger.info(f"\nAVAILABLE COLUMNS:")
        for col in first_model_df.columns:
            if col.endswith('_results'):
                logger.info(f"  {col}")
        
        logger.info(f"\n📁 All results saved to: {output_path}")
        logger.info(f"📄 Separate files created for each model:")
        for model_name in model_outputs.keys():
            logger.info(f"  - {model_name}_results.csv")
            logger.info(f"  - {model_name}_results.json")
            logger.info(f"  - {model_name}_outputs.txt")
        logger.info("\n" + "=" * 60)
        logger.info("🎉 REAL END-TO-END TEST COMPLETED!")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Real end-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        logger.info("\n🚀 Real end-to-end test successful!")
        logger.info("📝 Check the output directory for detailed results")
    else:
        logger.error("\n❌ Real end-to-end test failed.")
    
    sys.exit(0 if success else 1) 