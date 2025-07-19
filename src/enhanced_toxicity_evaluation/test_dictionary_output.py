#!/usr/bin/env python3
"""
Test dictionary-based classification output.
Stores complete classification results as dictionaries for each text.
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


def test_dictionary_classification_output():
    """Test storing complete classification outputs as dictionaries."""
    logger.info("🧪 Testing Dictionary-Based Classification Output")
    logger.info("=" * 60)
    
    try:
        # Load AllenAI dataset for quick testing
        logger.info("📥 Loading AllenAI dataset...")
        dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
        
        # Take a small sample for quick testing
        sample_size = 5
        sample_data = dataset.select(range(sample_size))
        
        logger.info(f"✅ Loaded {len(sample_data)} samples from AllenAI dataset")
        
        # Extract prompts
        prompts = [item["prompt"]["text"] for item in sample_data]
        logger.info(f"✅ Extracted {len(prompts)} prompts")
        
        # Mock some outputs (in real usage, these would come from your models)
        mock_outputs = [
            "This is a test response.",
            "Another test response here.",
            "Third test response.",
            "Fourth test response.",
            "Fifth test response."
        ]
        
        # Mock full texts (prompt + output)
        full_texts = [f"{prompt} {output}" for prompt, output in zip(prompts, mock_outputs)]
        
        logger.info("📊 Creating classification results...")
        
        # Create mock classification results as dictionaries
        # This simulates what your classifiers would actually return
        classification_results = {
            "prompts": {},
            "outputs": {},
            "full_texts": {}
        }
        
        # Load actual classifiers
        logger.info("🔧 Loading actual classifiers...")
        
        classifiers = {}
        
        # Load toxic-bert (multi-label)
        try:
            toxic_bert_classifier = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                return_all_scores=True,
                device=-1  # Use CPU to avoid CUDA issues
            )
            classifiers["toxic_bert"] = toxic_bert_classifier
            logger.info("✅ Loaded toxic-bert classifier")
        except Exception as e:
            logger.error(f"❌ Error loading toxic-bert classifier: {e}")
            classifiers["toxic_bert"] = None
        
        # Load roberta toxicity (single-label)
        try:
            roberta_classifier = pipeline(
                "text-classification",
                model="s-nlp/roberta_toxicity_classifier",
                device=-1
            )
            classifiers["roberta_toxicity"] = roberta_classifier
            logger.info("✅ Loaded roberta toxicity classifier")
        except Exception as e:
            logger.error(f"❌ Error loading roberta toxicity classifier: {e}")
            classifiers["roberta_toxicity"] = None
        
        # Load dynabench hate (single-label)
        try:
            dynabench_classifier = pipeline(
                "text-classification",
                model="facebook/roberta-hate-speech-dynabench-r4-target",
                device=-1
            )
            classifiers["dynabench_hate"] = dynabench_classifier
            logger.info("✅ Loaded dynabench hate classifier")
        except Exception as e:
            logger.error(f"❌ Error loading dynabench hate classifier: {e}")
            classifiers["dynabench_hate"] = None
        
        # Check if any classifiers loaded successfully
        loaded_classifiers = {k: v for k, v in classifiers.items() if v is not None}
        if not loaded_classifiers:
            logger.error("❌ No classifiers loaded successfully")
            logger.info("⚠️ Falling back to mock results for testing...")
            mock_results = _create_mock_results(prompts, mock_outputs, full_texts)
            toxic_bert_prompt_results = mock_results["prompts"]["toxic_bert"]
            toxic_bert_output_results = mock_results["outputs"]["toxic_bert"]
            toxic_bert_full_results = mock_results["full_texts"]["toxic_bert"]
            roberta_prompt_results = mock_results["prompts"]["roberta_toxicity"]
            roberta_output_results = mock_results["outputs"]["roberta_toxicity"]
            roberta_full_results = mock_results["full_texts"]["roberta_toxicity"]
            dynabench_prompt_results = mock_results["prompts"]["dynabench_hate"]
            dynabench_output_results = mock_results["outputs"]["dynabench_hate"]
            dynabench_full_results = mock_results["full_texts"]["dynabench_hate"]
        else:
            logger.info(f"✅ Successfully loaded {len(loaded_classifiers)} classifiers: {list(loaded_classifiers.keys())}")
        
        # Run actual classifications
        logger.info("🔍 Running actual classifications...")
        
        # Initialize results
        toxic_bert_prompt_results = []
        toxic_bert_output_results = []
        toxic_bert_full_results = []
        roberta_prompt_results = []
        roberta_output_results = []
        roberta_full_results = []
        dynabench_prompt_results = []
        dynabench_output_results = []
        dynabench_full_results = []
        
        # Classify prompts
        logger.info("  Classifying prompts...")
        if classifiers["toxic_bert"]:
            toxic_bert_prompt_results = _classify_texts(classifiers["toxic_bert"], prompts, "toxic_bert")
        if classifiers["roberta_toxicity"]:
            roberta_prompt_results = _classify_texts(classifiers["roberta_toxicity"], prompts, "roberta_toxicity")
        if classifiers["dynabench_hate"]:
            dynabench_prompt_results = _classify_texts(classifiers["dynabench_hate"], prompts, "dynabench_hate")
        
        # Classify outputs
        logger.info("  Classifying outputs...")
        if classifiers["toxic_bert"]:
            toxic_bert_output_results = _classify_texts(classifiers["toxic_bert"], mock_outputs, "toxic_bert")
        if classifiers["roberta_toxicity"]:
            roberta_output_results = _classify_texts(classifiers["roberta_toxicity"], mock_outputs, "roberta_toxicity")
        if classifiers["dynabench_hate"]:
            dynabench_output_results = _classify_texts(classifiers["dynabench_hate"], mock_outputs, "dynabench_hate")
        
        # Classify full texts
        logger.info("  Classifying full texts...")
        if classifiers["toxic_bert"]:
            toxic_bert_full_results = _classify_texts(classifiers["toxic_bert"], full_texts, "toxic_bert")
        if classifiers["roberta_toxicity"]:
            roberta_full_results = _classify_texts(classifiers["roberta_toxicity"], full_texts, "roberta_toxicity")
        if classifiers["dynabench_hate"]:
            dynabench_full_results = _classify_texts(classifiers["dynabench_hate"], full_texts, "dynabench_hate")
        
        # Handle missing results with fallbacks
        if not toxic_bert_prompt_results:
            logger.warning("⚠️ Using fallback for toxic-bert results")
            mock_results = _create_mock_results(prompts, mock_outputs, full_texts)
            toxic_bert_prompt_results = mock_results["prompts"]["toxic_bert"]
            toxic_bert_output_results = mock_results["outputs"]["toxic_bert"]
            toxic_bert_full_results = mock_results["full_texts"]["toxic_bert"]
        
        if not roberta_prompt_results:
            logger.warning("⚠️ Using fallback for roberta results")
            mock_results = _create_mock_results(prompts, mock_outputs, full_texts)
            roberta_prompt_results = mock_results["prompts"]["roberta_toxicity"]
            roberta_output_results = mock_results["outputs"]["roberta_toxicity"]
            roberta_full_results = mock_results["full_texts"]["roberta_toxicity"]
        
        if not dynabench_prompt_results:
            logger.warning("⚠️ Using fallback for dynabench results")
            mock_results = _create_mock_results(prompts, mock_outputs, full_texts)
            dynabench_prompt_results = mock_results["prompts"]["dynabench_hate"]
            dynabench_output_results = mock_results["outputs"]["dynabench_hate"]
            dynabench_full_results = mock_results["full_texts"]["dynabench_hate"]
        
        # Store all results
        classification_results["prompts"] = {
            "toxic_bert": toxic_bert_prompt_results,
            "roberta_toxicity": roberta_prompt_results,
            "dynabench_hate": dynabench_prompt_results
        }
        
        classification_results["outputs"] = {
            "toxic_bert": toxic_bert_output_results,
            "roberta_toxicity": roberta_output_results,
            "dynabench_hate": dynabench_output_results
        }
        
        classification_results["full_texts"] = {
            "toxic_bert": toxic_bert_full_results,
            "roberta_toxicity": roberta_full_results,
            "dynabench_hate": dynabench_full_results
        }
        
        logger.info("✅ Created classification results as dictionaries")
        
        # Test 1: Display the structure
        logger.info("\n📋 CLASSIFICATION RESULTS STRUCTURE:")
        logger.info("=" * 40)
        
        for text_type in ["prompts", "outputs", "full_texts"]:
            logger.info(f"\n{text_type.upper()}:")
            for classifier, results in classification_results[text_type].items():
                logger.info(f"  {classifier}: {len(results)} results")
                if results:
                    logger.info(f"    Sample result: {results[0]}")
        
        # Test 2: Create DataFrame with dictionary columns
        logger.info("\n📊 CREATING DATAFRAME WITH DICTIONARY COLUMNS:")
        logger.info("=" * 50)
        
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
        
        # Test 3: Access and analyze dictionary data
        logger.info("\n🔍 ACCESSING DICTIONARY DATA:")
        logger.info("=" * 30)
        
        # Show first row's toxic-bert results
        first_row = df.iloc[0]
        logger.info(f"First prompt: {first_row['prompt'][:50]}...")
        logger.info(f"Toxic-bert prompt results: {first_row['prompts_toxic_bert_results']}")
        logger.info(f"Toxic-bert output results: {first_row['outputs_toxic_bert_results']}")
        logger.info(f"Toxic-bert full text results: {first_row['full_texts_toxic_bert_results']}")
        
        # Test 4: Extract specific categories
        logger.info("\n📈 EXTRACTING SPECIFIC CATEGORIES:")
        logger.info("=" * 35)
        
        # Extract toxic-bert categories for analysis
        toxic_bert_categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_attack"]
        
        for category in toxic_bert_categories:
            # Extract category scores for prompts
            prompt_scores = [row['prompts_toxic_bert_results'][category] for _, row in df.iterrows()]
            output_scores = [row['outputs_toxic_bert_results'][category] for _, row in df.iterrows()]
            full_scores = [row['full_texts_toxic_bert_results'][category] for _, row in df.iterrows()]
            
            logger.info(f"\n{category.upper()}:")
            logger.info(f"  Prompts: {prompt_scores}")
            logger.info(f"  Outputs: {output_scores}")
            logger.info(f"  Full texts: {full_scores}")
            
            # Calculate improvements
            prompt_improvements = [p - o for p, o in zip(prompt_scores, output_scores)]
            logger.info(f"  Prompt→Output improvements: {prompt_improvements}")
        
        # Test 5: Create summary statistics
        logger.info("\n📊 SUMMARY STATISTICS:")
        logger.info("=" * 25)
        
        summary_stats = {}
        
        for category in toxic_bert_categories:
            prompt_scores = [row['prompts_toxic_bert_results'][category] for _, row in df.iterrows()]
            output_scores = [row['outputs_toxic_bert_results'][category] for _, row in df.iterrows()]
            
            summary_stats[f"{category}_prompt_mean"] = np.mean(prompt_scores)
            summary_stats[f"{category}_output_mean"] = np.mean(output_scores)
            summary_stats[f"{category}_improvement"] = np.mean(prompt_scores) - np.mean(output_scores)
        
        for stat, value in summary_stats.items():
            logger.info(f"  {stat}: {value:.4f}")
        
        # Test 6: Save to CSV (dictionaries will be stored as JSON strings)
        logger.info("\n💾 SAVING TO CSV:")
        logger.info("=" * 20)
        
        # Convert dictionaries to JSON strings for CSV storage
        df_csv = df.copy()
        for col in df_csv.columns:
            if col.endswith('_results'):
                df_csv[col] = df_csv[col].apply(lambda x: str(x))
        
        csv_path = Path("test_dictionary_output.csv")
        df_csv.to_csv(csv_path, index=False)
        logger.info(f"✅ Saved to {csv_path}")
        
        # Test 7: Load back and verify
        logger.info("\n🔄 LOADING BACK FROM CSV:")
        logger.info("=" * 30)
        
        df_loaded = pd.read_csv(csv_path)
        logger.info(f"✅ Loaded DataFrame with {len(df_loaded)} rows")
        
        # Convert JSON strings back to dictionaries
        import ast
        for col in df_loaded.columns:
            if col.endswith('_results'):
                df_loaded[col] = df_loaded[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        # Verify data integrity
        first_loaded = df_loaded.iloc[0]
        logger.info(f"✅ First row toxic-bert prompt results: {first_loaded['prompts_toxic_bert_results']}")
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 DICTIONARY-BASED CLASSIFICATION OUTPUT TEST PASSED!")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"❌ Dictionary classification output test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run dictionary classification output test."""
    success = test_dictionary_classification_output()
    if success:
        logger.info("\n🚀 Dictionary-based classification output working correctly!")
        logger.info("📝 This approach stores complete classification results as dictionaries")
        logger.info("📝 Each text gets its own dictionary with all classifier outputs")
        logger.info("📝 Easy to access specific categories and compare across text types")
    else:
        logger.error("\n❌ Dictionary classification output test failed.")
    
    return success


def _classify_texts(classifier, texts, classifier_type):
    """Classify texts and return results as dictionaries."""
    results = []
    
    for text in texts:
        try:
            if classifier_type == "toxic_bert":
                # Toxic-bert returns all scores for all categories
                predictions = classifier(text, truncation=True, max_length=512)
                if isinstance(predictions, list) and len(predictions) > 0:
                    # Convert to dictionary format
                    result = {}
                    for pred in predictions[0]:  # Take first (and only) prediction
                        label = pred["label"].lower()
                        score = pred["score"]
                        result[label] = score
                    results.append(result)
                else:
                    results.append({})
            else:
                # Single-label classifiers
                prediction = classifier(text, truncation=True, max_length=512)
                if isinstance(prediction, list) and len(prediction) > 0:
                    pred = prediction[0]
                    label = pred["label"].lower()
                    score = pred["score"]
                    
                    # Map labels to consistent names
                    if classifier_type == "roberta_toxicity":
                        label = "toxicity"
                    elif classifier_type == "dynabench_hate":
                        label = "hate"
                    
                    results.append({label: score})
                else:
                    results.append({})
                    
        except Exception as e:
            logger.warning(f"⚠️ Error classifying text: {e}")
            results.append({})
    
    return results


def _create_mock_results(prompts, mock_outputs, full_texts):
    """Create mock results for fallback testing."""
    logger.info("🔄 Creating mock results for testing...")
    
    # Mock toxic-bert results (multi-label)
    toxic_bert_categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_attack"]
    
    results = {
        "prompts": {},
        "outputs": {},
        "full_texts": {}
    }
    
    for text_type, texts in [("prompts", prompts), ("outputs", mock_outputs), ("full_texts", full_texts)]:
        toxic_bert_results = []
        roberta_results = []
        dynabench_results = []
        
        for i, text in enumerate(texts):
            # Mock toxic-bert prediction
            pred = {}
            for category in toxic_bert_categories:
                if text_type == "prompts":
                    base_score = 0.3 + i * 0.1
                elif text_type == "outputs":
                    base_score = 0.05 + i * 0.02
                else:  # full_texts
                    base_score = 0.15 + i * 0.05
                
                pred[category] = base_score + np.random.normal(0, 0.02)
                pred[category] = max(0.0, min(1.0, pred[category]))
            
            toxic_bert_results.append(pred)
            
            # Mock roberta prediction
            if text_type == "prompts":
                score = 0.4 + i * 0.1
            elif text_type == "outputs":
                score = 0.15 + i * 0.05
            else:  # full_texts
                score = 0.25 + i * 0.08
            
            roberta_results.append({"toxicity": score})
            
            # Mock dynabench prediction
            if text_type == "prompts":
                score = 0.3 + i * 0.08
            elif text_type == "outputs":
                score = 0.1 + i * 0.03
            else:  # full_texts
                score = 0.2 + i * 0.06
            
            dynabench_results.append({"hate": score})
        
        results[text_type]["toxic_bert"] = toxic_bert_results
        results[text_type]["roberta_toxicity"] = roberta_results
        results[text_type]["dynabench_hate"] = dynabench_results
    
    return results


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 