# RLHF Acceleration Guide

This guide provides step-by-step instructions to optimize RLHF training for maximum GPU utilization and speed.

## Quick Start

### 1. Run the Setup Script
```bash
python setup_acceleration.py
```

### 2. Start Training
```bash
# Basic accelerate training
./train_basic.sh

# DeepSpeed training (maximum optimization)
./train_deepspeed.sh
```

### 3. Monitor Performance
```bash
python monitor_gpu.py --interval 5 --output gpu_metrics.jsonl
```

## Expected Performance Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| GPU Utilization | 30% (4.4GB/15GB) | 90%+ (13.5GB/15GB) | 3x |
| Batch Processing | 90+ seconds | 5-10 seconds | 9-18x |
| mini_batch_size | 2 | 128 | 64x |
| Total Training Time | Days | Hours | 5-10x |

## Configuration Files

### Optimized Config: src/configs/rlhf/smolLM_135m_optimized.yaml
- Batch sizes: 1024 (vs 256)
- Mini batch: 128 (vs 32)
- PPO epochs: 2 (vs 4)
- Generation: Greedy decoding (faster)
- Mixed precision: FP16 enabled

### DeepSpeed Config: deepspeed_config.json
- ZeRO Stage 2: Memory optimization
- FP16: Mixed precision training
- Gradient clipping: 1.0
- Optimized communication: Overlap comm

## Training Commands

### Basic Accelerate Training
```bash
accelerate launch src/rlhf_train_optimized.py \
  rlhf=smolLM_135m_optimized \
  rlhf.model.batch_size=1024 \
  rlhf.model.mini_batch_size=128 \
  rlhf.model.ppo_epochs=2 \
  rlhf.output.organization=ajagota71
```

### DeepSpeed Training
```bash
accelerate launch --config_file deepspeed_config.json \
  src/rlhf_train_optimized.py \
  rlhf=smolLM_135m_optimized \
  rlhf.model.batch_size=1024 \
  rlhf.model.mini_batch_size=128 \
  rlhf.output.organization=ajagota71
```

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce batch sizes: `rlhf.model.batch_size=512`
- Enable CPU offload in DeepSpeed config
- Use gradient accumulation

### Config Overrides Not Working
- Always use `rlhf.model.*` prefix
- Check config with `OmegaConf.to_yaml(cfg)`

### Slow Training
- Monitor GPU utilization with `nvidia-smi`
- Verify batch sizes are applied correctly
- Enable DeepSpeed for maximum optimization

## Success Indicators

You'll know the optimization is working when:
- GPU memory usage is 90%+
- Batch processing time is 5-10 seconds
- No OOM errors occur
- Training completes in hours instead of days 