# Enhanced Toxicity Evaluation Pipeline

A comprehensive, modular pipeline for evaluating toxicity in language models with support for multiple models, classifiers, and evaluation strategies.

## 🚀 Quick Start

### 1. Test the Pipeline

First, test that everything works:

```bash
cd src/enhanced_toxicity_evaluation

# Start with the most basic test (recommended)
python minimal_test.py

# If that works, try the basic test
python basic_test.py

# Then try the simple test
python simple_test.py

# Or try the standard test
python test_pipeline.py

# If you encounter dataset issues, try the offline test
python offline_test.py
```

These will run minimal evaluations with a small model and dataset to verify the pipeline works correctly.

### 2. Run a Full Evaluation

```bash
# Run with default configuration
python run_evaluation.py

# Run with custom model configuration
python run_evaluation.py models=pythia_detox

# Run with different dataset
python run_evaluation.py dataset=real_toxicity_prompts max_prompts=500

# Run with lightweight evaluation
python run_evaluation.py classifiers=lightweight

# Combine multiple overrides
python run_evaluation.py models=pythia_detox dataset=real_toxicity_prompts max_prompts=1000
```

## 📁 Directory Structure

```
enhanced_toxicity_evaluation/
├── configs/                          # Hydra configuration files
│   ├── config.yaml                   # Main config with defaults
│   ├── models/                       # Model configurations
│   │   ├── pythia_detox.yaml         # Pythia detoxification checkpoints
│   │   └── rlhf_progression.yaml     # RLHF training progression
│   ├── dataset/                      # Dataset configurations
│   │   └── real_toxicity_prompts.yaml
│   ├── classifiers/                  # Toxicity classifier configs
│   │   ├── comprehensive.yaml        # All classifiers
│   │   └── lightweight.yaml          # Fast evaluation
│   ├── evaluation/                   # Evaluation strategy configs
│   │   └── standard.yaml             # Standard evaluation
│   └── output/                       # Output configuration
│       └── local_only.yaml           # Local storage only
├── core/                             # Core evaluation modules
│   ├── evaluator.py                  # Main evaluation engine
│   ├── model_loader.py               # Model loading utilities
│   ├── dataset_manager.py            # Dataset handling
│   ├── classifier_manager.py         # Toxicity classifier management
│   ├── generation_engine.py          # Text generation
│   └── metrics_calculator.py         # Metrics computation
├── run_evaluation.py                 # Main evaluation script
├── test_pipeline.py                  # Test script
└── README.md                         # This file
```

## ⚙️ Configuration

The pipeline uses Hydra for configuration management. Key configuration sections:

### Models Configuration

```yaml
# configs/models/pythia_detox.yaml
models:
  - name: "base"
    hf_path: "EleutherAI/pythia-410m"
    type: "base_model"
    
  - name: "epoch_20"
    hf_path: "ajagota71/pythia-410m-s-nlp-detox-checkpoint-epoch-20"
    type: "checkpoint"
    description: "Detoxified model at epoch 20"
```

### Dataset Configuration

```yaml
# configs/dataset/real_toxicity_prompts.yaml
dataset:
  name: "allenai/real-toxicity-prompts"
  split: "train"
  max_prompts: 1000
  filtering:
    strategy: "toxicity_threshold"
    toxicity_threshold: 0.4
    metrics: ["toxicity", "severe_toxicity"]
```

### Classifiers Configuration

```yaml
# configs/classifiers/comprehensive.yaml
classifiers:
  roberta_toxicity:
    enabled: true
    model: "s-nlp/roberta_toxicity_classifier"
    batch_size: 32
    max_length: 512
    
  toxic_bert:
    enabled: true
    model: "unitary/toxic-bert"
    batch_size: 32
    max_length: 512
```

## 🔧 Usage Examples

### Basic Evaluation

```bash
# Evaluate base model vs detoxified checkpoints
python run_evaluation.py models=pythia_detox dataset=real_toxicity_prompts
```

### RLHF Progression Analysis

```bash
# Evaluate RLHF training progression
python run_evaluation.py models=rlhf_progression max_prompts=2000
```

### Custom Configuration

```bash
# Use custom model paths
python run_evaluation.py \
  models.models.0.hf_path="your-org/your-model" \
  models.models.0.name="custom_model" \
  dataset.max_prompts=500 \
  generation.max_new_tokens=50
```

### Lightweight Evaluation

```bash
# Quick evaluation with minimal resources
python run_evaluation.py \
  classifiers=lightweight \
  dataset.max_prompts=100 \
  generation.batch_size=8
```

## 📊 Output

The pipeline generates comprehensive outputs:

### Files Generated

- `toxicity_evaluation_results.csv` - Main results with all scores and comparisons
- `evaluation_summary.json` - Summary metrics and statistics
- `raw_toxicity_results.json` - Raw classifier outputs
- `logs/evaluation.log` - Detailed execution log

### Key Metrics

- **Basic Statistics**: Mean, std, min, max toxicity scores
- **Distribution Metrics**: Percentiles, skewness, kurtosis
- **Threshold Metrics**: High/medium/low toxicity rates
- **Comparison Metrics**: Improvement vs baseline model
- **Statistical Tests**: Significance testing with p-values and effect sizes

### WandB Integration

If enabled, results are automatically logged to Weights & Biases with:
- Interactive plots and tables
- Model performance comparisons
- Statistical analysis results
- Sample outputs and prompts

## 🛠️ Customization

### Adding New Models

1. Create a new model configuration file in `configs/models/`
2. Define your models with HuggingFace paths
3. Run with your new configuration

### Adding New Classifiers

1. Add classifier configuration in `configs/classifiers/`
2. The pipeline automatically handles different classifier types
3. Supports binary and multi-label classifiers

### Custom Datasets

The pipeline supports multiple dataset strategies:
- `toxicity_threshold` - Filter by toxicity score
- `range` - Filter by toxicity range

## 🚨 Troubleshooting

### Dataset Loading Issues

If you encounter errors like `Invalid pattern: '**' can only be an entire path component` when loading the RealToxicityPrompts dataset, this is a known compatibility issue between different environments (Kaggle vs Colab). The pipeline includes multiple fallback strategies:

1. **Try the offline test first**:
```bash
python offline_test.py
```

2. **Use a smaller dataset**:
```bash
python run_evaluation.py dataset.max_prompts=10
```

3. **The pipeline will automatically try**:
   - Direct loading
   - Loading with specific revision
   - Loading with explicit data files
   - Manual download from HuggingFace Hub
   - Fallback to test prompts

### Memory Issues

If you encounter CUDA out-of-memory errors:

1. **Reduce batch sizes**:
```bash
python run_evaluation.py generation.batch_size=2 classifiers.evaluation.batch_size=4
```

2. **Use smaller models**:
```bash
python run_evaluation.py models.models.0.hf_path="EleutherAI/pythia-70m"
```

3. **Reduce sequence lengths**:
```bash
python run_evaluation.py generation.max_length=256 classifiers.classifiers.roberta_toxicity.max_length=128
```

### Model Loading Issues

If models fail to load:

1. **Check internet connection** (for downloading models)
2. **Verify model paths** are correct
3. **Try different torch_dtype settings**:
```bash
python run_evaluation.py models.model_loading.torch_dtype=float16
```
- `top_k` - Select top-k most toxic prompts
- `random` - Random sampling
- `cached` - Load from cached dataset
- `custom` - Load from custom file
- `generated` - Generate new dataset

## 🔍 Analysis Tools

### Results Inspection

```python
import pandas as pd

# Load results
df = pd.read_csv("results/toxicity_eval_20231201_120000/toxicity_evaluation_results.csv")

# View model comparisons
comparison_cols = [col for col in df.columns if col.startswith('delta_')]
print(df[comparison_cols].describe())

# Find most improved examples
best_improvements = df.nlargest(10, 'delta_epoch_100_vs_base_roberta_toxicity_score')
print(best_improvements[['prompt', 'output_base', 'output_epoch_100']])
```

### Statistical Analysis

The pipeline automatically performs:
- Wilcoxon signed-rank tests
- Paired t-tests
- Mann-Whitney U tests
- Cohen's d effect size calculations
- Multiple comparison corrections

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch sizes or use smaller models
   ```bash
   python run_evaluation.py generation.batch_size=4 models=small_models
   ```

2. **Model Loading Failures**: Check HuggingFace paths and model availability
   ```bash
   python run_evaluation.py models.model_loading.trust_remote_code=true
   ```

3. **Classifier Errors**: Enable fallback to safe scores
   ```bash
   python run_evaluation.py classifiers.error_handling.fallback_to_safe=true
   ```

### Debug Mode

Enable detailed logging:
```bash
python run_evaluation.py logging.log_level=DEBUG
```

### Test Mode

Run minimal evaluation for testing:
```bash
python test_pipeline.py
```

## 📈 Performance Optimization

### For Large-Scale Evaluation

1. **Use Caching**: Enable dataset caching to avoid re-downloading
2. **Parallel Processing**: Use multiple workers for classifier evaluation
3. **Batch Processing**: Optimize batch sizes for your hardware
4. **Model Quantization**: Use 8-bit or 4-bit quantization for large models

### Memory Management

- The pipeline automatically handles model cleanup
- Uses adaptive batch sizing for OOM recovery
- Supports CPU fallback for large models

## 🤝 Contributing

To extend the pipeline:

1. **Add New Components**: Create new modules in the `core/` directory
2. **Configuration**: Add corresponding config files in `configs/`
3. **Testing**: Update `test_pipeline.py` to test new features
4. **Documentation**: Update this README with new features

## 📝 License

This pipeline is part of the IRL LLMs project. See the main project license for details. 