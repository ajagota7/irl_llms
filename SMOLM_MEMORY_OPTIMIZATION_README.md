# SmolLM-1.7B Memory Optimization Guide

This guide explains how to run SmolLM-1.7B RLHF training without running out of memory on an A100 GPU.

## Problem

SmolLM-1.7B is a 1.7 billion parameter model that requires significant GPU memory for RLHF training. The original configuration was causing CUDA out of memory errors on A100 GPUs.

## Solutions

We've created three optimized configurations with different memory usage levels:

### 1. Standard Optimized (`smolLM_1p7b.yaml`)
- **Batch size**: 4 (reduced from 8)
- **Mini batch size**: 1 (reduced from 2)
- **Forward batch size**: 1 (reduced from 2)
- **PPO epochs**: 2 (reduced from 4)
- **Generation length**: 10-15 tokens (reduced from 15-20)
- **Effective batch size**: 4 (maintained through gradient accumulation)

### 2. Conservative (`smolLM_1p7b_conservative.yaml`)
- **Batch size**: 2
- **Mini batch size**: 1
- **Forward batch size**: 1
- **PPO epochs**: 1
- **Generation length**: 8-12 tokens
- **Effective batch size**: 2 (with gradient accumulation)

### 3. Advanced Memory Optimization (`smolLM_1p7b_memory_optimized.yaml`)
- **Batch size**: 2
- **Mini batch size**: 1
- **Forward batch size**: 1
- **PPO epochs**: 1
- **Generation length**: 8-12 tokens
- **Additional optimizations**:
  - Half precision (FP16/BF16)
  - Gradient checkpointing
  - Automatic mixed precision (AMP)
  - Low CPU memory usage
  - Disabled KV cache during generation
  - Expandable segments for better memory management

## Usage

### Option 1: Use the helper script
```bash
python run_smolLM_optimized.py
```

### Option 2: Direct command line

#### Standard optimized (recommended first try):
```bash
python src/rlhf_train.py rlhf=smolLM_1p7b model.use_raw_logits=true model.reward_model="s-nlp/roberta_toxicity_classifier" training.save_freq=50 training.num_train_epochs=100 output.checkpoint_push_freq=20 output.push_to_hub=true output.organization=ajagota71
```

#### Conservative settings:
```bash
python src/rlhf_train.py rlhf=smolLM_1p7b_conservative model.use_raw_logits=true model.reward_model="s-nlp/roberta_toxicity_classifier" training.save_freq=50 training.num_train_epochs=100 output.checkpoint_push_freq=20 output.push_to_hub=true output.organization=ajagota71
```

#### Advanced memory optimization:
```bash
python src/rlhf_train.py rlhf=smolLM_1p7b_memory_optimized model.use_raw_logits=true model.reward_model="s-nlp/roberta_toxicity_classifier" training.save_freq=50 training.num_train_epochs=100 output.checkpoint_push_freq=20 output.push_to_hub=true output.organization=ajagota71
```

## Memory Optimization Techniques

### 1. Reduced Batch Sizes
- Smaller batch sizes reduce peak memory usage
- Gradient accumulation maintains effective batch size
- Trade-off: Slightly slower training

### 2. Half Precision
- Uses FP16 or BF16 instead of FP32
- Reduces memory usage by ~50%
- BF16 is preferred for Ampere+ GPUs (A100, H100)

### 3. Gradient Checkpointing
- Trades computation for memory
- Recomputes intermediate activations instead of storing them
- Reduces memory usage by ~30-50%

### 4. Generation Optimizations
- Shorter generation lengths reduce memory during inference
- Disabled KV cache saves memory during generation
- Reduced PPO epochs decrease memory spikes

### 5. Environment Variables
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for better memory fragmentation handling

## Troubleshooting

### Still getting OOM errors?

1. **Try the conservative configuration first**
2. **Further reduce batch size**: Add `model.batch_size=1` to your command
3. **Reduce generation length**: Add `model.generation.output_max_length=8`
4. **Clear GPU cache**: Restart your Python session
5. **Check for other processes**: Ensure no other GPU processes are running

### Performance vs Memory Trade-offs

| Configuration | Memory Usage | Training Speed | Quality |
|---------------|--------------|----------------|---------|
| Original | High | Fast | Best |
| Standard Optimized | Medium | Medium | Good |
| Conservative | Low | Slow | Good |
| Advanced Optimized | Low | Medium | Good |

## Expected Memory Usage

- **Original config**: ~40GB+ (causes OOM on A100)
- **Standard optimized**: ~25-30GB
- **Conservative**: ~15-20GB
- **Advanced optimized**: ~12-18GB

## Monitoring Memory Usage

You can monitor GPU memory usage during training:
```python
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
```

## Recommendations

1. **Start with standard optimized** - it should work on most A100 setups
2. **If OOM occurs**, switch to conservative configuration
3. **For maximum memory efficiency**, use advanced memory optimization
4. **Monitor memory usage** during the first few epochs
5. **Consider reducing batch size further** if still having issues

## Customization

You can create your own configuration by copying one of the existing configs and modifying the parameters. Key parameters to adjust:

- `model.batch_size`: Main batch size
- `model.mini_batch_size`: Mini-batch size for PPO
- `model.gradient_accumulation_steps`: Steps for gradient accumulation
- `model.generation.output_max_length`: Maximum generation length
- `model.ppo_epochs`: Number of PPO epochs per step 