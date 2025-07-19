"""
Toxicity classifier management utilities for the Enhanced Toxicity Evaluation Pipeline.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from omegaconf import DictConfig
from transformers import pipeline
import torch
import numpy as np

logger = logging.getLogger(__name__)


class ClassifierManager:
    """Manages multiple toxicity classifiers with parallel processing and error handling."""
    
    def __init__(self, config: DictConfig):
        """Initialize the classifier manager with configuration."""
        self.config = config
        self.classifiers_config = config.get("classifiers", {})
        self.evaluation_config = config.get("evaluation", {})
        self.output_config = config.get("output_format", {})
        self.error_config = config.get("error_handling", {})
        
        self.classifiers = {}
        self.device = self._setup_device()
        
        logger.info(f"ClassifierManager initialized with device: {self.device}")
    
    def _setup_device(self) -> str:
        """Setup device for classifiers."""
        if torch.cuda.is_available():
            return "cuda:0"
        else:
            return "cpu"
    
    def _is_multi_label_classifier(self, model_name: str) -> bool:
        """Detect if a classifier returns multi-label results."""
        multi_label_models = [
            "toxic-bert", "unitary/toxic-bert",
            "toxigen", "detoxify"
        ]
        return any(name in model_name.lower() for name in multi_label_models)

    def _get_toxic_bert_categories(self) -> List[str]:
        """Get standard toxic-bert category names."""
        return ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_attack"]
    
    def load_classifiers(self) -> Dict[str, Any]:
        """Load all enabled classifiers."""
        logger.info("Loading toxicity classifiers...")
        
        for name, config in self.classifiers_config.items():
            if not config.get("enabled", True):
                logger.info(f"Skipping disabled classifier: {name}")
                continue
            
            try:
                classifier = self._load_single_classifier(name, config)
                if classifier:
                    self.classifiers[name] = classifier
                    logger.info(f"✅ Loaded classifier: {name}")
                else:
                    logger.warning(f"Failed to load classifier: {name}")
                    
            except Exception as e:
                logger.error(f"❌ Error loading classifier {name}: {e}")
                if not self.error_config.get("skip_failed_classifiers", True):
                    raise
        
        logger.info(f"Loaded {len(self.classifiers)} classifiers successfully")
        return self.classifiers
    
    def _load_single_classifier(self, name: str, config: Dict[str, Any]) -> Optional[Any]:
        """Load a single classifier with error handling."""
        model_name = config["model"]
        batch_size = config.get("batch_size", 32)
        max_length = config.get("max_length", 512)
        
        try:
            if self._is_multi_label_classifier(model_name):
                # Special handling for multi-label classifiers like toxic-bert
                classifier = pipeline(
                    "text-classification",
                    model=model_name,
                    device=0 if self.device.startswith("cuda") else -1,
                    truncation=True,
                    max_length=max_length,
                    return_all_scores=True
                )
            else:
                # Standard binary classifier
                classifier = pipeline(
                    "text-classification",
                    model=model_name,
                    device=0 if self.device.startswith("cuda") else -1,
                    truncation=True,
                    max_length=max_length
                )
            
            return {
                "pipeline": classifier,
                "config": config,
                "name": name
            }
            
        except Exception as e:
            logger.error(f"Failed to load classifier {name}: {e}")
            return None
    
    def evaluate_texts(self, texts: List[str], text_type: str = "output") -> Dict[str, List[Dict]]:
        """Evaluate toxicity of texts using all loaded classifiers."""
        logger.info(f"Evaluating toxicity on {len(texts)} {text_type}s using {len(self.classifiers)} classifiers")
        
        if not self.classifiers:
            logger.warning("No classifiers loaded, loading them now...")
            self.load_classifiers()
        
        results = {}
        
        if self.evaluation_config.get("parallel", True):
            results = self._evaluate_parallel(texts)
        else:
            results = self._evaluate_sequential(texts)
        
        logger.info(f"Completed toxicity evaluation for {len(texts)} texts")
        return results
    
    def _evaluate_parallel(self, texts: List[str]) -> Dict[str, List[Dict]]:
        """Evaluate texts using parallel processing."""
        max_workers = self.evaluation_config.get("max_workers", 4)
        timeout = self.evaluation_config.get("timeout", 300)
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit evaluation tasks
            future_to_classifier = {}
            for name, classifier_info in self.classifiers.items():
                future = executor.submit(
                    self._evaluate_with_classifier,
                    name,
                    classifier_info,
                    texts
                )
                future_to_classifier[future] = name
            
            # Collect results
            for future in as_completed(future_to_classifier, timeout=timeout):
                classifier_name = future_to_classifier[future]
                try:
                    classifier_results = future.result()
                    results[classifier_name] = classifier_results
                    logger.info(f"Completed evaluation for {classifier_name}")
                except Exception as e:
                    logger.error(f"Error evaluating with {classifier_name}: {e}")
                    if self.error_config.get("fallback_to_safe", True):
                        results[classifier_name] = self._create_safe_fallback(len(texts))
        
        return results
    
    def _evaluate_sequential(self, texts: List[str]) -> Dict[str, List[Dict]]:
        """Evaluate texts sequentially."""
        results = {}
        
        for name, classifier_info in self.classifiers.items():
            try:
                classifier_results = self._evaluate_with_classifier(name, classifier_info, texts)
                results[name] = classifier_results
                logger.info(f"Completed evaluation for {name}")
            except Exception as e:
                logger.error(f"Error evaluating with {name}: {e}")
                if self.error_config.get("fallback_to_safe", True):
                    results[name] = self._create_safe_fallback(len(texts))
        
        return results
    
    def _evaluate_with_classifier(self, name: str, classifier_info: Dict, texts: List[str]) -> List[Dict]:
        """Evaluate texts with a specific classifier."""
        classifier = classifier_info["pipeline"]
        config = classifier_info["config"]
        batch_size = config.get("batch_size", 32)
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                if name == "toxic_bert":
                    # Handle toxic-bert's multiple labels
                    predictions = classifier(batch)
                    
                    # Process toxic-bert results - preserve original list format for multi-label extraction
                    for pred in predictions:
                        if isinstance(pred, list):
                            # Multiple categories returned - keep as list for extract_detailed_scores
                            results.append(pred)
                        else:
                            # Single prediction - convert to list format
                            results.append([pred])
                else:
                    # Handle standard binary classifiers
                    predictions = classifier(batch)
                    
                    for pred in predictions:
                        if isinstance(pred, list):
                            # Multiple labels returned
                            toxic_score = 0.0
                            for item in pred:
                                label = item["label"].lower()
                                if any(word in label for word in ["toxic", "hate", "harassment"]):
                                    toxic_score = max(toxic_score, item["score"])
                            results.append({"score": toxic_score})
                        else:
                            # Single label
                            label = pred["label"].lower()
                            score = pred["score"]
                            
                            # Determine if this is a toxic prediction
                            if any(word in label for word in ["toxic", "hate", "harassment"]):
                                results.append({"score": score})
                            else:
                                # Invert score for non-toxic labels
                                results.append({"score": 1.0 - score})
                                
            except Exception as e:
                logger.error(f"Error in batch {i} for classifier {name}: {e}")
                
                # Add fallback results for this batch
                fallback_results = self._create_safe_fallback(len(batch))
                results.extend(fallback_results)
        
        return results
    
    def _create_safe_fallback(self, num_texts: int) -> List[Dict]:
        """Create safe fallback results when classifier fails."""
        return [{"score": 0.0, "error": "classifier_failed"} for _ in range(num_texts)]
    
    def extract_detailed_scores(self, predictions: Dict[str, List[Dict]], text_type: str = "text") -> Dict[str, List[float]]:
        """Extract detailed toxicity scores from classifier predictions with multi-label support."""
        detailed_scores = {}
        
        if not predictions:
            return detailed_scores
        
        num_texts = len(next(iter(predictions.values())))
        
        for classifier_name, classifier_predictions in predictions.items():
            if self._is_multi_label_classifier(classifier_name):
                # Handle multi-label classifiers (toxic-bert)
                categories = self._get_toxic_bert_categories()
                
                for category in categories:
                    scores = []
                    for pred in classifier_predictions:
                        if isinstance(pred, list):
                            # Find the category score in the list
                            category_score = 0.0
                            for item in pred:
                                if item["label"].lower() == category.lower():
                                    category_score = item["score"]
                                    break
                            scores.append(category_score)
                        elif isinstance(pred, dict):
                            # Dictionary format (from _evaluate_with_classifier)
                            category_score = pred.get(category.lower(), 0.0)
                            scores.append(category_score)
                        else:
                            # Single prediction case - handle different formats
                            if isinstance(pred, dict):
                                if "label" in pred and pred["label"].lower() == category.lower():
                                    scores.append(pred["score"])
                                else:
                                    scores.append(0.0)
                            else:
                                scores.append(0.0)
                    
                    detailed_scores[f"{classifier_name}_{category}"] = scores
            else:
                # Handle binary classifiers
                scores = []
                for pred in classifier_predictions:
                    if isinstance(pred, list):
                        # Multiple labels - find the toxic one
                        toxic_score = 0.0
                        for item in pred:
                            label = item["label"].lower()
                            if any(word in label for word in ["toxic", "hate", "harassment"]):
                                toxic_score = max(toxic_score, item["score"])
                        scores.append(toxic_score)
                    elif isinstance(pred, dict) and "score" in pred:
                        # Dictionary format with score
                        scores.append(pred["score"])
                    else:
                        # Single label - handle different formats
                        if isinstance(pred, dict) and "label" in pred:
                            label = pred["label"].lower()
                            if any(word in label for word in ["toxic", "hate", "harassment"]):
                                scores.append(pred["score"])
                            else:
                                scores.append(1.0 - pred["score"])
                        else:
                            scores.append(0.0)
                
                detailed_scores[f"{classifier_name}_score"] = scores
        
        return detailed_scores
    
    def get_classifier_info(self) -> Dict[str, Any]:
        """Get information about loaded classifiers."""
        info = {}
        for name, classifier_info in self.classifiers.items():
            info[name] = {
                "model": classifier_info["config"]["model"],
                "description": classifier_info["config"].get("description", ""),
                "batch_size": classifier_info["config"].get("batch_size", 32),
                "max_length": classifier_info["config"].get("max_length", 512)
            }
        return info
    
    def cleanup(self):
        """Clean up loaded classifiers to free memory."""
        for name, classifier_info in self.classifiers.items():
            logger.info(f"Cleaning up classifier: {name}")
            del classifier_info["pipeline"]
        
        self.classifiers.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Classifier cleanup completed") 