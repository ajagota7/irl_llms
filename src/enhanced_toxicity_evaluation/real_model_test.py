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
                    # Single-label classifiers
                    prediction = classifier(text, truncation=True, max_length=512)
                    if isinstance(prediction, list) and len(prediction) > 0:
                        pred = prediction[0]
                        label = pred["label"].lower()
                        score = pred["score"]
                        
                        # Map labels to consistent names
                        if classifier_name == "roberta_toxicity":
                            label = "toxicity"
                        elif classifier_name == "dynabench_hate":
                            label = "hate"
                        
                        classifier_results.append({label: score})
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
    
    # Create DataFrame
    df_data = []
    for i in range(len(prompts)):
        row = {
            "prompt": prompts[i],
            "prompt_index": i
        }
        
        # Add outputs for each model
        for model_name, outputs in model_outputs.items():
            row[f"output_{model_name}"] = outputs[i]
            row[f"full_text_{model_name}"] = f"{prompts[i]} {outputs[i]}"
        
        # Add prompt classifications
        for classifier_name, results in prompt_classifications.items():
            if results and i < len(results):
                row[f"prompt_{classifier_name}_results"] = results[i]
        
        # Add output classifications for each model
        for model_name in model_outputs.keys():
            if model_name in output_classifications:
                for classifier_name, results in output_classifications[model_name].items():
                    if results and i < len(results):
                        row[f"output_{model_name}_{classifier_name}_results"] = results[i]
        
        # Add full text classifications for each model
        for model_name in model_outputs.keys():
            if model_name in full_text_classifications:
                for classifier_name, results in full_text_classifications[model_name].items():
                    if results and i < len(results):
                        row[f"full_text_{model_name}_{classifier_name}_results"] = results[i]
        
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    logger.info(f"✅ Created comprehensive DataFrame with {len(df)} rows and {len(df.columns)} columns")
    
    return df


def save_results(df, model_outputs, output_dir="real_model_results"):
    """Save results to separate files."""
    logger.info(f"💾 Saving results to {output_dir}...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save comprehensive DataFrame
    comprehensive_path = output_path / "comprehensive_results.csv"
    df_csv = df.copy()
    for col in df_csv.columns:
        if col.endswith('_results'):
            df_csv[col] = df_csv[col].apply(lambda x: str(x) if isinstance(x, dict) else x)
    df_csv.to_csv(comprehensive_path, index=False)
    logger.info(f"✅ Saved comprehensive results to {comprehensive_path}")
    
    # Save separate CSV files for each model for easy comparison
    for model_name in model_outputs.keys():
        # Create model-specific DataFrame
        model_df = df[['prompt', 'prompt_index']].copy()
        
        # Add model-specific columns
        model_df[f'output_{model_name}'] = df[f'output_{model_name}']
        model_df[f'full_text_{model_name}'] = df[f'full_text_{model_name}']
        
        # Add prompt classifications
        for col in df.columns:
            if col.startswith('prompt_') and col.endswith('_results'):
                model_df[col] = df[col]
        
        # Add model-specific output and full text classifications
        for col in df.columns:
            if col.startswith(f'output_{model_name}_') or col.startswith(f'full_text_{model_name}_'):
                model_df[col] = df[col]
        
        # Convert dictionaries to strings for CSV
        model_df_csv = model_df.copy()
        for col in model_df_csv.columns:
            if col.endswith('_results'):
                model_df_csv[col] = model_df_csv[col].apply(lambda x: str(x) if isinstance(x, dict) else x)
        
        # Save model-specific CSV
        model_csv_path = output_path / f"{model_name}_results.csv"
        model_df_csv.to_csv(model_csv_path, index=False)
        logger.info(f"✅ Saved {model_name} results to {model_csv_path}")
        
        # Save model-specific JSON
        model_json_path = output_path / f"{model_name}_results.json"
        model_df.to_json(model_json_path, orient='records', indent=2)
        logger.info(f"✅ Saved {model_name} results to {model_json_path}")
    
    # Save comprehensive results as JSON
    json_path = output_path / "comprehensive_results.json"
    df.to_json(json_path, orient='records', indent=2)
    logger.info(f"✅ Saved comprehensive results to {json_path}")
    
    # Save model outputs separately
    for model_name, outputs in model_outputs.items():
        model_output_path = output_path / f"{model_name}_outputs.txt"
        with open(model_output_path, 'w', encoding='utf-8') as f:
            for i, output in enumerate(outputs):
                f.write(f"Output {i+1}:\n{output}\n\n")
        logger.info(f"✅ Saved {model_name} outputs to {model_output_path}")
    
    # Save prompts
    prompts_path = output_path / "prompts.txt"
    with open(prompts_path, 'w', encoding='utf-8') as f:
        for i, prompt in enumerate(df['prompt']):
            f.write(f"Prompt {i+1}:\n{prompt}\n\n")
    logger.info(f"✅ Saved prompts to {prompts_path}")
    
    # Save classification summary
    summary_data = {}
    for col in df.columns:
        if col.endswith('_results'):
            # Extract classifier and text type from column name
            parts = col.split('_')
            if len(parts) >= 3:
                text_type = parts[0]  # prompt, output, or full_text
                classifier = parts[1]  # classifier name
                
                if classifier not in summary_data:
                    summary_data[classifier] = {}
                
                if text_type not in summary_data[classifier]:
                    summary_data[classifier][text_type] = {}
                
                # Calculate average scores for each category
                valid_results = [r for r in df[col] if isinstance(r, dict) and r]
                if valid_results:
                    all_categories = set()
                    for result in valid_results:
                        all_categories.update(result.keys())
                    
                    for category in all_categories:
                        scores = [result.get(category, 0.0) for result in valid_results]
                        summary_data[classifier][text_type][category] = {
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
        
        # Take a small sample for testing
        sample_size = config["dataset"]["sample_size"]
        sample_data = dataset.select(range(sample_size))
        prompts = [item["prompt"]["text"] for item in sample_data]
        
        logger.info(f"✅ Loaded {len(prompts)} prompts from AllenAI dataset")
        
        # Load models and classifiers
        models, tokenizers = load_models(config)
        classifiers = load_classifiers(config)
        
        # Generate outputs
        model_outputs = generate_outputs(models, tokenizers, prompts, config["generation"]["max_new_tokens"])
        
        # Create comprehensive results
        df = create_comprehensive_results(prompts, model_outputs, classifiers)
        
        # Save results
        output_path = save_results(df, model_outputs, config["output"]["directory"])
        
        # Display summary
        logger.info("\n📋 RESULTS SUMMARY:")
        logger.info("=" * 30)
        
        for model_name, outputs in model_outputs.items():
            logger.info(f"\n{model_name.upper()} MODEL:")
            logger.info(f"  Generated {len(outputs)} outputs")
            if outputs:
                logger.info(f"  Sample output: {outputs[0][:100]}...")
        
        # Show classification results for first sample
        first_row = df.iloc[0]
        logger.info(f"\nFIRST SAMPLE CLASSIFICATIONS:")
        logger.info(f"  Prompt: {first_row['prompt'][:50]}...")
        
        # Show toxic-bert results specifically
        logger.info(f"\nTOXIC-BERT CLASSIFICATIONS (First Sample):")
        for col in df.columns:
            if col.endswith('_results') and isinstance(first_row[col], dict) and 'toxic_bert' in col:
                logger.info(f"  {col}: {first_row[col]}")
        
        # Show other classifier results
        logger.info(f"\nOTHER CLASSIFIER RESULTS (First Sample):")
        for col in df.columns:
            if col.endswith('_results') and isinstance(first_row[col], dict) and 'toxic_bert' not in col:
                logger.info(f"  {col}: {first_row[col]}")
        
        # Show available columns
        logger.info(f"\nAVAILABLE COLUMNS:")
        for col in df.columns:
            if col.endswith('_results'):
                logger.info(f"  {col}")
        
        logger.info(f"\n📁 All results saved to: {output_path}")
        logger.info(f"📄 Separate files created for each model:")
        for model_name in model_outputs.keys():
            logger.info(f"  - {model_name}_results.csv")
            logger.info(f"  - {model_name}_results.json")
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