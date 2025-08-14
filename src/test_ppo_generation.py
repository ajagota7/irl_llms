"""
Test script to verify PPO generation format is correct.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model

def test_ppo_generation_format():
    """Test that PPO generation works with the correct tensor format."""
    
    print("Testing PPO generation format...")
    
    # Load a small model for testing
    model_name = "HuggingFaceTB/SmolLM-135M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    ref_model = create_reference_model(model)
    
    # Create a simple PPO config
    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=1e-6,
        batch_size=4,
        mini_batch_size=2,
        gradient_accumulation_steps=2,
        log_with=None
    )
    
    # Create PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=[],  # Empty dataset for testing
    )
    
    # Test queries
    test_queries = [
        "This is a test query.",
        "Another test query here."
    ]
    
    print("Testing generation with individual tensors...")
    
    try:
        # Test the correct format (individual tensors)
        for i, query in enumerate(test_queries):
            query_tensor = tokenizer(query, return_tensors="pt")
            query_input_ids = query_tensor.input_ids.squeeze()
            
            print(f"  Query {i}: {query} -> Length: {len(query_input_ids)}")
            
            # Generate with the correct format
            generation_kwargs = {
                "min_length": 5,
                "max_new_tokens": 10,
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id,
            }
            
            response = ppo_trainer.generate(query_input_ids, **generation_kwargs)
            response_text = tokenizer.decode(response, skip_special_tokens=True)
            print(f"  Response: {response_text}")
        
        print("\n✅ SUCCESS: PPO generation works with individual tensors!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_ppo_generation_format()
    if success:
        print("\n🎉 PPO generation format test passed!")
    else:
        print("\n💥 PPO generation format test failed.") 