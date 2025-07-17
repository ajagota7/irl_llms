"""
Model loading utilities for the Enhanced Toxicity Evaluation Pipeline.
"""

import os
import torch
import logging
from typing import Dict, List, Optional, Any
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    hf_path: str
    model_type: str
    description: str
    device: str
    dtype: str


class ModelLoader:
    """Handles loading of language models with proper error handling and fallback strategies."""
    
    def __init__(self, config: DictConfig):
        """Initialize the model loader with configuration."""
        self.config = config
        self.models = {}
        self.tokenizers = {}
        self.device = self._setup_device()
        self.loading_settings = config.get("model_loading", {})
        self.fallback_settings = config.get("fallback_settings", {})
        
        logger.info(f"ModelLoader initialized with device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup and configure device."""
        device_config = self.config.get("device", "auto")
        
        if device_config == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_config)
        
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
        return device
    
    def _determine_model_size(self, model_name: str) -> str:
        """Determine model size category for fallback settings."""
        model_name_lower = model_name.lower()
        
        # Large models (> 1B parameters)
        if any(size in model_name_lower for size in ["1b", "2.7b", "6b", "7b", "12b", "70b", "large"]):
            return "large_models"
        
        # Medium models (100M-1B parameters)
        if any(size in model_name_lower for size in ["410m", "160m", "125m", "70m"]):
            return "medium_models"
        
        # Small models (< 100M parameters)
        return "small_models"
    
    def _get_loading_kwargs(self, model_name: str) -> Dict[str, Any]:
        """Get loading keyword arguments based on model size and configuration."""
        model_size = self._determine_model_size(model_name)
        fallback_config = self.fallback_settings.get(model_size, {})
        
        # Start with default settings
        kwargs = {
            "device_map": self.loading_settings.get("device_map", "auto"),
            "trust_remote_code": self.loading_settings.get("trust_remote_code", True),
            "low_cpu_mem_usage": self.loading_settings.get("low_cpu_mem_usage", True),
        }
        
        # Handle torch_dtype
        torch_dtype = self.loading_settings.get("torch_dtype", "auto")
        if torch_dtype == "auto":
            # Use fallback settings based on model size
            if model_size == "large_models":
                kwargs["torch_dtype"] = torch.bfloat16
            elif model_size == "medium_models":
                kwargs["torch_dtype"] = torch.float16
            else:
                kwargs["torch_dtype"] = torch.float32
        else:
            kwargs["torch_dtype"] = torch_dtype
        
        # Handle quantization
        if fallback_config.get("load_in_8bit", False):
            kwargs["load_in_8bit"] = True
        elif fallback_config.get("load_in_4bit", False):
            kwargs["load_in_4bit"] = True
        
        return kwargs
    
    def load_single_model(self, model_config: Dict[str, Any]) -> Optional[ModelInfo]:
        """Load a single model with error handling and fallback strategies."""
        model_name = model_config["name"]
        hf_path = model_config["hf_path"]
        model_type = model_config.get("type", "unknown")
        description = model_config.get("description", "")
        
        logger.info(f"Loading model: {model_name} from {hf_path}")
        
        try:
            # Get loading parameters
            loading_kwargs = self._get_loading_kwargs(hf_path)
            
            # Load tokenizer first
            logger.info(f"Loading tokenizer for {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(hf_path)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
            
            # Load model with fallback strategies
            logger.info(f"Loading model {model_name}...")
            model = self._load_model_with_fallback(hf_path, loading_kwargs)
            
            # Create model info
            model_info = ModelInfo(
                name=model_name,
                model=model,
                tokenizer=tokenizer,
                hf_path=hf_path,
                model_type=model_type,
                description=description,
                device=str(self.device),
                dtype=str(loading_kwargs.get("torch_dtype", "unknown"))
            )
            
            logger.info(f"✅ Successfully loaded {model_name}")
            return model_info
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_name}: {e}")
            return None
    
    def _load_model_with_fallback(self, hf_path: str, loading_kwargs: Dict[str, Any]) -> AutoModelForCausalLM:
        """Load model with multiple fallback strategies."""
        fallback_strategies = [
            # Strategy 1: Original settings
            lambda: AutoModelForCausalLM.from_pretrained(hf_path, **loading_kwargs),
            
            # Strategy 2: Remove device_map for CPU fallback
            lambda: AutoModelForCausalLM.from_pretrained(
                hf_path, 
                **{k: v for k, v in loading_kwargs.items() if k != "device_map"}
            ),
            
            # Strategy 3: Remove quantization
            lambda: AutoModelForCausalLM.from_pretrained(
                hf_path,
                **{k: v for k, v in loading_kwargs.items() 
                   if k not in ["load_in_8bit", "load_in_4bit"]}
            ),
            
            # Strategy 4: Minimal settings
            lambda: AutoModelForCausalLM.from_pretrained(
                hf_path,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ),
        ]
        
        for i, strategy in enumerate(fallback_strategies):
            try:
                logger.info(f"Trying loading strategy {i+1} for {hf_path}")
                model = strategy()
                return model
            except Exception as e:
                logger.warning(f"Strategy {i+1} failed: {e}")
                if i == len(fallback_strategies) - 1:
                    raise e
                continue
    
    def load_all_models(self) -> Dict[str, ModelInfo]:
        """Load all models specified in the configuration."""
        models_config = self.config.get("models", [])
        
        if not models_config:
            logger.warning("No models specified in configuration")
            return {}
        
        loaded_models = {}
        
        for model_config in models_config:
            model_info = self.load_single_model(model_config)
            if model_info:
                loaded_models[model_info.name] = model_info
                self.models[model_info.name] = model_info.model
                self.tokenizers[model_info.name] = model_info.tokenizer
        
        logger.info(f"Loaded {len(loaded_models)} models successfully")
        return loaded_models
    
    def get_model(self, model_name: str) -> Optional[AutoModelForCausalLM]:
        """Get a loaded model by name."""
        return self.models.get(model_name)
    
    def get_tokenizer(self, model_name: str) -> Optional[AutoTokenizer]:
        """Get a loaded tokenizer by name."""
        return self.tokenizers.get(model_name)
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get model information by name."""
        for model_info in self.models.values():
            if hasattr(model_info, 'name') and model_info.name == model_name:
                return model_info
        return None
    
    def cleanup(self):
        """Clean up loaded models to free memory."""
        for model_name, model in self.models.items():
            logger.info(f"Cleaning up model: {model_name}")
            del model
        
        for model_name, tokenizer in self.tokenizers.items():
            logger.info(f"Cleaning up tokenizer: {model_name}")
            del tokenizer
        
        self.models.clear()
        self.tokenizers.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model cleanup completed") 