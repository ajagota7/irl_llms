# Gemma3 Model Training Fixes

## Problem Description

When training Gemma3 models with RLHF, you may encounter TorchDynamo compilation errors like:

```
torch._dynamo.exc.Unsupported: Logger not supported for non-export cases. To avoid graph breaks caused by logger in compile-mode, it is recommended to disable logging by adding logging methods to config.ignore_logger_methods
```

This happens because:
1. Gemma3 models use logging during forward passes that TorchDynamo can't handle
2. The default attention implementation (`sdpa`) can cause issues with Gemma3 models
3. TorchDynamo compilation is enabled by default in newer PyTorch versions

## Solutions Implemented

### 1. Updated Configuration Files

The following configuration parameters have been added to fix the issues:

```yaml
model:
  # Model loading parameters to fix TorchDynamo issues
  attn_implementation: "eager"  # Recommended for Gemma3 models
  torch_compile: false  # Disable TorchDynamo compilation
  use_cache: true
```

### 2. Updated Training Script

The `rlhf_train.py` script now:
- Sets environment variables to completely disable TorchDynamo
- Disables TorchDynamo compilation globally
- Uses the `eager` attention implementation for Gemma3 models
- Passes the configuration parameters to model loading
- Wraps generation calls with `torch._dynamo.disable()`

### 3. Conservative Configuration

A conservative configuration (`gemma_270m_conservative.yaml`) is provided with:
- Smaller batch sizes (32 instead of 64)
- Lower learning rate (5e-7 instead of 1e-6)
- Fewer PPO epochs (2 instead of 4)
- Greedy decoding instead of sampling
- Shorter generation lengths

### 4. Wrapper Script

A wrapper script (`run_gemma_rlhf.py`) is provided that:
- Sets all necessary environment variables
- Runs the training with TorchDynamo completely disabled
- Provides a simple interface for running training

## Usage

### Option 1: Use the Wrapper Script (Recommended)

```bash
# Run with conservative configuration (default)
python run_gemma_rlhf.py

# Run with original configuration
python run_gemma_rlhf.py rlhf=gemma_270m model.use_raw_logits=true model.reward_model=s-nlp/roberta_toxicity_classifier training.save_freq=50 training.num_train_epochs=100 output.checkpoint_push_freq=20 output.push_to_hub=true output.organization=your_org

# Run with conservative configuration
python run_gemma_rlhf.py rlhf=gemma_270m_conservative model.use_raw_logits=true model.reward_model=s-nlp/roberta_toxicity_classifier output.push_to_hub=true output.organization=your_org
```

### Option 2: Use the Fixed Training Script Directly

```bash
python src/rlhf_train.py rlhf=gemma_270m model.use_raw_logits=true model.reward_model=s-nlp/roberta_toxicity_classifier training.save_freq=50 training.num_train_epochs=100 output.checkpoint_push_freq=20 output.push_to_hub=true output.organization=your_org
```

### Option 3: Test TorchDynamo Disable First

```bash
python test_torchdynamo_disable.py rlhf=gemma_270m
```

## Key Changes Made

1. **Environment Variables**: Set multiple environment variables to completely disable TorchDynamo
2. **Global TorchDynamo Disable**: Added `torch._dynamo.config.disable = True` to completely disable compilation
3. **Generation Wrapping**: Wrapped generation calls with `torch._dynamo.disable()` context manager
4. **Eager Attention**: Using `attn_implementation="eager"` as recommended for Gemma3 models
5. **Model Loading Parameters**: Added support for configuration-based model loading parameters
6. **Conservative Alternative**: Created a safer configuration for initial testing
7. **Wrapper Script**: Created a script that sets all environment variables before running training

## Expected Behavior

After these fixes:
- The model should load without TorchDynamo compilation errors
- Training should proceed normally
- You should see the warning about using `eager` attention (which is expected and good)
- The model should train successfully with the specified parameters

## Troubleshooting

If you still encounter issues:

1. **Memory Issues**: Try the conservative configuration with smaller batch sizes
2. **CUDA Errors**: Ensure you have enough GPU memory for the batch size
3. **Generation Issues**: The conservative config uses greedy decoding which is more stable
4. **Slow Training**: The eager attention implementation may be slightly slower but more stable

## Additional Notes

- The `eager` attention implementation is specifically recommended for Gemma3 models by the Hugging Face team
- Disabling TorchDynamo compilation may slightly reduce training speed but increases stability
- The conservative configuration is designed for initial testing and can be adjusted once you confirm everything works 