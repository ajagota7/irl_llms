"""
Text generation engine for the Enhanced Toxicity Evaluation Pipeline.
"""

import torch
import logging
from typing import Dict, List, Optional, Any
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GenerationEngine:
    """Handles text generation from multiple language models."""
    
    def __init__(self, config: DictConfig):
        """Initialize the generation engine with configuration."""
        self.config = config
        self.generation_params = config.get("generation", {})
        self.device = self._setup_device()
        
        logger.info(f"GenerationEngine initialized with device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup and configure device."""
        device_config = self.config.get("device", "auto")
        
        if device_config == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_config)
        
        return device
    
    def generate_all(self, models: Dict[str, Any], prompts: List[str]) -> Dict[str, List[str]]:
        """Generate completions for all models."""
        logger.info(f"Generating completions for {len(models)} models on {len(prompts)} prompts")
        
        all_outputs = {}
        
        for model_name, model_info in models.items():
            logger.info(f"Generating with model: {model_name}")
            
            try:
                outputs = self._generate_with_model(model_info, prompts)
                all_outputs[model_name] = outputs
                logger.info(f"✅ Generated {len(outputs)} outputs for {model_name}")
                
            except Exception as e:
                logger.error(f"❌ Error generating with {model_name}: {e}")
                # Add empty outputs for failed model
                all_outputs[model_name] = [""] * len(prompts)
        
        return all_outputs
    
    def _generate_with_model(self, model_info: Any, prompts: List[str]) -> List[str]:
        """Generate completions with a specific model."""
        model = model_info.model
        tokenizer = model_info.tokenizer
        
        # Set model to evaluation mode
        model.eval()
        
        batch_size = self.generation_params.get("batch_size", 16)
        outputs = []
        
        # Process prompts in batches
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generating with {model_info.name}"):
            batch_prompts = prompts[i:i + batch_size]
            
            try:
                batch_outputs = self._generate_batch(model, tokenizer, batch_prompts)
                outputs.extend(batch_outputs)
                
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM error with batch size {batch_size}, reducing...")
                # Reduce batch size and retry
                new_batch_size = max(1, batch_size // 2)
                if new_batch_size != batch_size:
                    batch_size = new_batch_size
                    # Retry with smaller batch
                    batch_outputs = self._generate_batch(model, tokenizer, batch_prompts, batch_size)
                    outputs.extend(batch_outputs)
                else:
                    # Add empty outputs if we can't reduce further
                    outputs.extend([""] * len(batch_prompts))
                    
            except Exception as e:
                logger.error(f"Error generating batch {i}: {e}")
                # Add empty outputs for failed batch
                outputs.extend([""] * len(batch_prompts))
        
        return outputs
    
    def _generate_batch(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                       prompts: List[str], batch_size: Optional[int] = None) -> List[str]:
        """Generate completions for a batch of prompts."""
        if batch_size is None:
            batch_size = self.generation_params.get("batch_size", 16)
        
        # Tokenize the batch
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.generation_params.get("max_length", 1024)
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate completions
        with torch.no_grad():
            generated_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.generation_params.get("max_new_tokens", 100),
                do_sample=self.generation_params.get("do_sample", True),
                temperature=self.generation_params.get("temperature", 0.7),
                top_p=self.generation_params.get("top_p", 1.0),
                use_cache=self.generation_params.get("use_cache", True),
                pad_token_id=self.generation_params.get("pad_token_id", tokenizer.eos_token_id),
            )
        
        # Extract only the generated tokens (remove the prompt)
        batch_outputs = []
        for j, generated_id in enumerate(generated_ids):
            # Find where the prompt ends
            prompt_length = inputs["input_ids"][j].shape[0]
            
            # Extract only the new tokens
            new_tokens = generated_id[prompt_length:]
            
            # Decode the new tokens
            output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            batch_outputs.append(output_text.strip())
        
        return batch_outputs
    
    def generate_full_texts(self, models: Dict[str, Any], prompts: List[str], 
                           existing_completions: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[str]]:
        """Generate full texts (prompt + completion) for all models."""
        logger.info("Generating full texts for all models")
        
        # Use existing completions if provided, otherwise generate new ones
        if existing_completions is None:
            completions = self.generate_all(models, prompts)
        else:
            completions = existing_completions
        
        # Combine prompts with completions
        full_texts = {}
        for model_name, model_completions in completions.items():
            full_texts[model_name] = [
                f"{prompt} {completion}".strip() 
                for prompt, completion in zip(prompts, model_completions)
            ]
        
        return full_texts
    
    def get_generation_info(self) -> Dict[str, Any]:
        """Get information about the generation configuration."""
        return {
            "max_new_tokens": self.generation_params.get("max_new_tokens", 100),
            "temperature": self.generation_params.get("temperature", 0.7),
            "top_p": self.generation_params.get("top_p", 1.0),
            "do_sample": self.generation_params.get("do_sample", True),
            "batch_size": self.generation_params.get("batch_size", 16),
            "max_length": self.generation_params.get("max_length", 1024),
            "device": str(self.device)
        } 