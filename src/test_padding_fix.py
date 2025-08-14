"""
Test script to verify the padding fix for batched generation.
"""

import torch
from transformers import AutoTokenizer

def test_padding_fix():
    """Test that tensors of different sizes can be properly padded and stacked."""
    
    print("Testing padding fix for batched generation...")
    
    # Simulate a tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Simulate queries of different lengths
    queries = [
        "This is a short query.",
        "This is a much longer query that will have more tokens when tokenized.",
        "Medium length query here."
    ]
    
    # Tokenize queries
    batch_queries = []
    max_length = 0
    
    print("Original queries:")
    for i, query in enumerate(queries):
        query_tensor = tokenizer(query, return_tensors="pt")
        query_input_ids = query_tensor.input_ids.squeeze()
        print(f"  Query {i}: {query} -> Length: {len(query_input_ids)}")
        max_length = max(max_length, len(query_input_ids))
        batch_queries.append(query_input_ids)
    
    print(f"\nMaximum length: {max_length}")
    
    # Pad all queries to the same length
    padded_queries = []
    for query_input_ids in batch_queries:
        if len(query_input_ids) < max_length:
            # Pad with pad_token_id
            padding_length = max_length - len(query_input_ids)
            padding = torch.full((padding_length,), tokenizer.pad_token_id)
            padded_query = torch.cat([query_input_ids, padding], dim=0)
        else:
            padded_query = query_input_ids
        padded_queries.append(padded_query)
        print(f"  Padded length: {len(padded_query)}")
    
    # Stack queries for batched generation
    try:
        stacked_queries = torch.stack(padded_queries)
        print(f"\n✅ SUCCESS: Stacked tensor shape: {stacked_queries.shape}")
        print("All tensors are now the same size and can be processed in batch!")
        return True
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_padding_fix()
    if success:
        print("\n🎉 Padding fix test passed! The batched generation should work now.")
    else:
        print("\n💥 Padding fix test failed. There's still an issue to resolve.") 