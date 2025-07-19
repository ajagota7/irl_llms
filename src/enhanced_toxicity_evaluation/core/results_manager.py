"""
Results management utilities for the Enhanced Toxicity Evaluation Pipeline.
Handles creation and organization of comprehensive evaluation results.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from omegaconf import DictConfig
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ResultsManager:
    """Manages creation and organization of comprehensive evaluation results."""
    
    def __init__(self, config: DictConfig):
        """Initialize the results manager with configuration."""
        self.config = config
        self.output_config = config.get("output", {})
        
        logger.info("ResultsManager initialized")
    
    def create_comprehensive_results(self, prompts: List[str], 
                                   model_outputs: Dict[str, List[str]], 
                                   classifiers: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Create comprehensive results with all classifications."""
        logger.info("📊 Creating comprehensive results...")
        
        # Classify prompts
        from .classifier_manager import ClassifierManager
        classifier_manager = ClassifierManager(self.config)
        classifier_manager.classifiers = classifiers
        
        prompt_classifications = classifier_manager.classify_texts(prompts, "prompts")
        
        # Classify outputs for each model
        output_classifications = {}
        full_text_classifications = {}
        
        for model_name, outputs in model_outputs.items():
            # Classify outputs
            output_classifications[model_name] = classifier_manager.classify_texts(outputs, f"outputs_{model_name}")
            
            # Create full texts (prompt + output)
            full_texts = [f"{prompt} {output}" for prompt, output in zip(prompts, outputs)]
            full_text_classifications[model_name] = classifier_manager.classify_texts(full_texts, f"full_texts_{model_name}")
        
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
    
    def save_results(self, model_dfs: Dict[str, pd.DataFrame], 
                    model_outputs: Dict[str, List[str]], 
                    output_dir: str = "real_model_results") -> Path:
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
    
    def print_classifier_summary(self, df: pd.DataFrame):
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
    
    def get_results_info(self) -> Dict[str, Any]:
        """Get information about the results configuration."""
        return {
            "output_directory": self.output_config.get("directory", "real_model_results"),
            "save_formats": self.output_config.get("save_formats", ["csv", "json", "txt"]),
            "include_summary": self.output_config.get("include_summary", True)
        } 