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
    
    def load_classifiers(self) -> Dict[str, Any]:
        """Load toxicity classifiers with comprehensive error handling."""
        logger.info("🔧 Loading toxicity classifiers...")
        
        for classifier_name, classifier_config in self.classifiers_config.items():
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
                
                self.classifiers[classifier_name] = classifier
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
                    
                    self.classifiers[classifier_name] = classifier
                    logger.info(f"✅ Loaded {classifier_name} classifier on CPU")
                    
                except Exception as e2:
                    logger.error(f"❌ Failed to load {classifier_name} even on CPU: {e2}")
                    self.classifiers[classifier_name] = None
        
        return self.classifiers
    
    def classify_texts(self, texts: List[str], text_type: str) -> Dict[str, List[Dict]]:
        """Classify texts and return results as dictionaries."""
        logger.info(f"🔍 Classifying {text_type}...")
        
        results = {}
        
        for classifier_name, classifier in self.classifiers.items():
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
    
    def load_classifiers_legacy(self) -> Dict[str, Any]:
        """Load all enabled classifiers (legacy method)."""
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
                            # Convert to dictionary format
                            result = {}
                            for label_pred in pred:
                                label = label_pred["label"].lower()
                                score = label_pred["score"]
                                result[label] = score
                            results.append(result)
                        else:
                            results.append({})
                else:
                    # Handle single-label classifiers
                    predictions = classifier(batch, return_all_scores=True)
                    
                    for pred in predictions:
                        if isinstance(pred, list):
                            # Convert to dictionary format
                            result = {}
                            for label_pred in pred:
                                label = label_pred["label"].lower()
                                score = label_pred["score"]
                                result[label] = score
                            results.append(result)
                        else:
                            results.append({})
                            
            except Exception as e:
                logger.error(f"Error evaluating batch {i} with {name}: {e}")
                # Add empty results for failed batch
                results.extend([{}] * len(batch))
        
        return results
    
    def _create_safe_fallback(self, num_texts: int) -> List[Dict]:
        """Create safe fallback results when classifier fails."""
        return [{"safe": 1.0, "unsafe": 0.0} for _ in range(num_texts)]
    
    def extract_detailed_scores(self, predictions: Dict[str, List[Dict]], text_type: str = "text") -> Dict[str, List[float]]:
        """Extract detailed scores from classifier predictions."""
        detailed_scores = {}
        
        for classifier_name, classifier_predictions in predictions.items():
            if classifier_name == "toxic_bert":
                # Extract all toxic-bert categories
                categories = self._get_toxic_bert_categories()
                for category in categories:
                    scores = []
                    for pred in classifier_predictions:
                        if isinstance(pred, dict) and category in pred:
                            scores.append(pred[category])
                        else:
                            scores.append(0.0)
                    detailed_scores[f"{text_type}_{classifier_name}_{category}"] = scores
            else:
                # Extract main toxicity score for other classifiers
                main_score_key = "toxic" if "toxic" in classifier_name else "hate"
                scores = []
                for pred in classifier_predictions:
                    if isinstance(pred, dict) and main_score_key in pred:
                        scores.append(pred[main_score_key])
                    else:
                        scores.append(0.0)
                detailed_scores[f"{text_type}_{classifier_name}"] = scores
        
        return detailed_scores
    
    def get_classifier_info(self) -> Dict[str, Any]:
        """Get information about loaded classifiers."""
        info = {}
        for name, classifier in self.classifiers.items():
            if classifier is not None:
                info[name] = {
                    "loaded": True,
                    "device": str(self.device),
                    "type": "multi_label" if self._is_multi_label_classifier(name) else "single_label"
                }
            else:
                info[name] = {"loaded": False}
        return info
    
    def cleanup(self):
        """Clean up loaded classifiers."""
        logger.info("Cleaning up classifier manager...")
        
        for name, classifier in self.classifiers.items():
            if classifier is not None:
                del classifier
        
        self.classifiers.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Classifier manager cleanup completed") 