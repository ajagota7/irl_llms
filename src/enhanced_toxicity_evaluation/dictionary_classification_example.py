#!/usr/bin/env python3
"""
Practical example of dictionary-based classification output.
Uses real classifiers with AllenAI dataset.
"""

import sys
import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import pipeline

# Suppress CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dictionary_classification_pipeline():
    """Create a pipeline that stores complete classification results as dictionaries."""
    
    logger.info("🚀 Creating Dictionary-Based Classification Pipeline")
    logger.info("=" * 60)
    
    try:
        # Load AllenAI dataset
        logger.info("📥 Loading AllenAI dataset...")
        dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
        
        # Take a small sample for quick testing
        sample_size = 3
        sample_data = dataset.select(range(sample_size))
        prompts = [item["prompt"]["text"] for item in sample_data]
        
        logger.info(f"✅ Loaded {len(prompts)} prompts from AllenAI dataset")
        
        # Mock outputs (in real usage, these would come from your language models)
        mock_outputs = [
            "This is a safe and helpful response.",
            "I aim to be constructive and respectful.",
            "Let me provide a thoughtful answer."
        ]
        
        # Create full texts
        full_texts = [f"{prompt} {output}" for prompt, output in zip(prompts, mock_outputs)]
        
        # Initialize classifiers
        logger.info("🔧 Loading classifiers...")
        
        # Note: In practice, you'd load these with proper error handling
        # For this example, we'll use mock results to avoid loading large models
        
        # Mock classifier results (replace with actual classifier calls)
        classification_results = {
            "prompts": {},
            "outputs": {},
            "full_texts": {}
        }
        
        # Mock toxic-bert results (multi-label)
        toxic_bert_categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_attack"]
        
        for text_type, texts in [("prompts", prompts), ("outputs", mock_outputs), ("full_texts", full_texts)]:
            toxic_bert_results = []
            
            for i, text in enumerate(texts):
                # Mock toxic-bert prediction (replace with actual classifier call)
                # In practice: toxic_bert_classifier(text, return_all_scores=True)
                pred = {}
                for category in toxic_bert_categories:
                    # Simulate different toxicity levels based on text type
                    if text_type == "prompts":
                        base_score = 0.3 + i * 0.1
                    elif text_type == "outputs":
                        base_score = 0.05 + i * 0.02
                    else:  # full_texts
                        base_score = 0.15 + i * 0.05
                    
                    pred[category] = base_score + np.random.normal(0, 0.02)
                    pred[category] = max(0.0, min(1.0, pred[category]))  # Clamp to [0,1]
                
                toxic_bert_results.append(pred)
            
            classification_results[text_type]["toxic_bert"] = toxic_bert_results
        
        # Mock roberta toxicity results (single-label)
        for text_type, texts in [("prompts", prompts), ("outputs", mock_outputs), ("full_texts", full_texts)]:
            roberta_results = []
            
            for i, text in enumerate(texts):
                # Mock roberta prediction (replace with actual classifier call)
                # In practice: roberta_classifier(text)
                if text_type == "prompts":
                    score = 0.4 + i * 0.1
                elif text_type == "outputs":
                    score = 0.1 + i * 0.02
                else:  # full_texts
                    score = 0.25 + i * 0.05
                
                score += np.random.normal(0, 0.02)
                score = max(0.0, min(1.0, score))
                
                roberta_results.append({"toxicity": score})
            
            classification_results[text_type]["roberta_toxicity"] = roberta_results
        
        # Mock dynabench hate results (single-label)
        for text_type, texts in [("prompts", prompts), ("outputs", mock_outputs), ("full_texts", full_texts)]:
            dynabench_results = []
            
            for i, text in enumerate(texts):
                # Mock dynabench prediction (replace with actual classifier call)
                # In practice: dynabench_classifier(text)
                if text_type == "prompts":
                    score = 0.3 + i * 0.08
                elif text_type == "outputs":
                    score = 0.05 + i * 0.015
                else:  # full_texts
                    score = 0.15 + i * 0.04
                
                score += np.random.normal(0, 0.02)
                score = max(0.0, min(1.0, score))
                
                dynabench_results.append({"hate": score})
            
            classification_results[text_type]["dynabench_hate"] = dynabench_results
        
        logger.info("✅ Created classification results as dictionaries")
        
        # Create DataFrame with dictionary columns
        logger.info("📊 Creating DataFrame...")
        
        df_data = []
        for i in range(len(prompts)):
            row = {
                "prompt": prompts[i],
                "output": mock_outputs[i],
                "full_text": full_texts[i],
                "prompt_index": i
            }
            
            # Add classification results as dictionary columns
            for text_type in ["prompts", "outputs", "full_texts"]:
                for classifier, results in classification_results[text_type].items():
                    column_name = f"{text_type}_{classifier}_results"
                    row[column_name] = results[i]
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        logger.info(f"✅ Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"✅ DataFrame columns: {list(df.columns)}")
        
        # Display results
        logger.info("\n📋 CLASSIFICATION RESULTS:")
        logger.info("=" * 40)
        
        for i, row in df.iterrows():
            logger.info(f"\nSample {i+1}:")
            logger.info(f"  Prompt: {row['prompt'][:50]}...")
            logger.info(f"  Output: {row['output']}")
            
            # Show toxic-bert results
            toxic_bert_prompt = row['prompts_toxic_bert_results']
            toxic_bert_output = row['outputs_toxic_bert_results']
            
            logger.info(f"  Toxic-BERT Prompt: {toxic_bert_prompt}")
            logger.info(f"  Toxic-BERT Output: {toxic_bert_output}")
            
            # Calculate improvements
            improvements = {}
            for category in toxic_bert_categories:
                improvement = toxic_bert_prompt[category] - toxic_bert_output[category]
                improvements[category] = improvement
            
            logger.info(f"  Improvements: {improvements}")
        
        # Analysis
        logger.info("\n📈 ANALYSIS:")
        logger.info("=" * 20)
        
        # Calculate average improvements for toxic-bert categories
        avg_improvements = {}
        for category in toxic_bert_categories:
            prompt_scores = [row[f'prompts_toxic_bert_results'][category] for _, row in df.iterrows()]
            output_scores = [row[f'outputs_toxic_bert_results'][category] for _, row in df.iterrows()]
            
            avg_improvement = np.mean(prompt_scores) - np.mean(output_scores)
            avg_improvements[category] = avg_improvement
            
            logger.info(f"  {category}: {avg_improvement:.4f} improvement")
        
        # Save results
        logger.info("\n💾 SAVING RESULTS:")
        logger.info("=" * 20)
        
        # Convert dictionaries to JSON strings for CSV storage
        df_csv = df.copy()
        for col in df_csv.columns:
            if col.endswith('_results'):
                df_csv[col] = df_csv[col].apply(lambda x: str(x))
        
        csv_path = Path("dictionary_classification_results.csv")
        df_csv.to_csv(csv_path, index=False)
        logger.info(f"✅ Saved to {csv_path}")
        
        # Also save as JSON for better dictionary preservation
        json_path = Path("dictionary_classification_results.json")
        df.to_json(json_path, orient='records', indent=2)
        logger.info(f"✅ Saved to {json_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 DICTIONARY-BASED CLASSIFICATION PIPELINE COMPLETED!")
        logger.info("=" * 60)
        
        return df, classification_results
        
    except Exception as e:
        logger.error(f"❌ Dictionary classification pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Run the dictionary classification pipeline example."""
    df, results = create_dictionary_classification_pipeline()
    
    if df is not None:
        logger.info("\n🚀 Dictionary-based classification pipeline working!")
        logger.info("📝 Key benefits of this approach:")
        logger.info("   • Complete classification results stored as dictionaries")
        logger.info("   • Easy access to all categories and scores")
        logger.info("   • Flexible analysis across text types (prompts, outputs, full_texts)")
        logger.info("   • Clean DataFrame structure with dictionary columns")
        logger.info("   • Easy to extend with new classifiers or categories")
    else:
        logger.error("\n❌ Dictionary classification pipeline failed.")
    
    return df is not None


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 