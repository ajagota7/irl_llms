# Modular Enhanced Toxicity Evaluation Pipeline

This document describes the modular architecture of the Enhanced Toxicity Evaluation Pipeline, which has been refactored from the monolithic `real_model_test.py` into separate, reusable components.

## Architecture Overview

The pipeline is now organized into two main phases:

1. **Classification Phase**: Loads models, generates text, and classifies everything
2. **Evaluation Phase**: Creates visualizations and insights from classified data

This separation allows you to:
- Run classification once and evaluate multiple times
- Track metrics across different inference processes
- Reuse classified data for different analysis approaches
- Scale each phase independently

## Core Components

### 1. ModelLoader (`core/model_loader.py`)
- Loads language models with comprehensive error handling
- Supports multiple models (base, detoxified, etc.)
- Handles GPU/CPU fallback strategies
- Manages model memory efficiently

### 2. ClassifierManager (`core/classifier_manager.py`)
- Loads toxicity classifiers (Toxic-BERT, RoBERTa, Dynabench)
- Handles multi-label and single-label classifiers
- Provides parallel processing capabilities
- Manages classifier fallback strategies

### 3. GenerationEngine (`core/generation_engine.py`)
- Generates text from loaded models
- Handles batch processing and memory management
- Supports various generation parameters
- Provides progress tracking

### 4. DatasetManager (`core/dataset_manager.py`)
- Loads and processes datasets (RealToxicityPrompts)
- Supports filtering by toxicity thresholds
- Handles caching for performance
- Provides dataset statistics

### 5. ResultsManager (`core/results_manager.py`)
- Creates comprehensive results from all classifications
- Organizes data into structured formats
- Saves results in multiple formats (CSV, JSON, TXT)
- Generates summary statistics

### 6. EvaluationPipeline (`core/evaluation_pipeline.py`)
- Orchestrates the complete pipeline
- Separates classification and evaluation phases
- Provides flexible execution options
- Handles error recovery

### 7. VisualizationManager (`core/visualization_manager.py`)
- Creates comprehensive visualizations
- Supports interactive plots with Plotly
- Integrates with Weights & Biases
- Generates statistical analysis

## Configuration

The modular architecture uses a comprehensive configuration system:

```yaml
# models: Define language models to evaluate
models:
  base:
    path: "EleutherAI/pythia-70m"
  detox_epoch_20:
    path: "ajagota71/pythia-70m-s-nlp-detox-checkpoint-epoch-20"

# classifiers: Define toxicity classifiers
classifiers:
  toxic_bert:
    model: "unitary/toxic-bert"
    return_all_scores: true
    device: 0

# generation: Text generation parameters
generation:
  max_new_tokens: 50
  temperature: 0.7

# dataset: Dataset configuration
dataset:
  name: "allenai/real-toxicity-prompts"
  max_prompts: 100
  filtering:
    strategy: "toxicity_threshold"
    toxicity_threshold: 0.5
```

## Usage Examples

### 1. Run Classification Phase Only

```python
from core import EvaluationPipeline
from omegaconf import OmegaConf
import yaml

# Load configuration
with open("configs/modular_config.yaml", 'r') as f:
    config_dict = yaml.safe_load(f)
config = OmegaConf.create(config_dict)

# Create pipeline
pipeline = EvaluationPipeline(config)

# Run classification phase
result = pipeline.run_classification_phase()

if result["success"]:
    print(f"Results saved to: {result['output_path']}")
```

### 2. Run Evaluation Phase Only (Using Existing Results)

```python
# Run evaluation phase using existing results
result = pipeline.run_evaluation_phase(results_path="modular_results")

if result["success"]:
    print(f"Visualizations saved to: {result['output_path']}")
```

### 3. Run Full Pipeline

```python
# Run complete pipeline
result = pipeline.run_full_pipeline()

if result["success"]:
    print("Pipeline completed successfully!")
```

### 4. Test Individual Components

```python
from core import ModelLoader, ClassifierManager, DatasetManager

# Test model loading
model_loader = ModelLoader(config)
models, tokenizers = model_loader.load_models()
print(f"Loaded {len(models)} models")

# Test classifier loading
classifier_manager = ClassifierManager(config)
classifiers = classifier_manager.load_classifiers()
print(f"Loaded {len(classifiers)} classifiers")

# Test dataset loading
dataset_manager = DatasetManager(config)
prompts = dataset_manager.get_prompts()
print(f"Loaded {len(prompts)} prompts")
```

## Test File

The `test_modular_pipeline.py` file provides comprehensive testing:

```bash
# Run all tests
python test_modular_pipeline.py

# The test file includes:
# 1. Individual component testing
# 2. Classification phase testing
# 3. Evaluation phase testing
# 4. Full pipeline testing
```

## Output Structure

The modular pipeline creates organized output:

```
modular_results/
├── base_results.csv              # Base model results
├── detox_epoch_20_results.csv    # Detoxified model results
├── comprehensive_results.csv     # Combined results
├── comprehensive_results.json    # JSON format
├── classification_summary.json   # Statistical summary
├── model_comparison.json        # Model comparison
├── base_outputs.txt             # Raw model outputs
├── prompts.txt                  # Input prompts
├── plots/                       # Generated visualizations
│   ├── toxicity_reduction_*.png
│   ├── prompt_comparison_*.png
│   └── interactive_*.html
└── model_mapping.json           # Model metadata
```

## Key Benefits

### 1. Separation of Concerns
- **Classification**: Focus on model loading, text generation, and toxicity classification
- **Evaluation**: Focus on analysis, visualization, and insights

### 2. Reusability
- Run classification once, evaluate multiple times
- Reuse classified data for different analysis approaches
- Mix and match components as needed

### 3. Scalability
- Scale classification and evaluation independently
- Add new models or classifiers easily
- Extend visualization capabilities

### 4. Maintainability
- Clear component boundaries
- Comprehensive error handling
- Detailed logging and debugging

### 5. Flexibility
- Run phases separately or together
- Customize configuration for different use cases
- Easy to extend with new features

## Migration from real_model_test.py

The modular architecture preserves all functionality from `real_model_test.py`:

| real_model_test.py Function | Modular Component |
|-----------------------------|-------------------|
| `load_config()` | Configuration loading in test files |
| `load_models()` | `ModelLoader.load_models()` |
| `load_classifiers()` | `ClassifierManager.load_classifiers()` |
| `generate_outputs()` | `GenerationEngine.generate_outputs()` |
| `classify_texts()` | `ClassifierManager.classify_texts()` |
| `create_comprehensive_results()` | `ResultsManager.create_comprehensive_results()` |
| `save_results()` | `ResultsManager.save_results()` |
| `create_toxicity_plots()` | `EvaluationPipeline._create_toxicity_plots()` |
| `create_interactive_plots()` | `EvaluationPipeline._create_interactive_plots()` |

## Configuration Files

- `configs/modular_config.yaml`: Main configuration for modular pipeline
- `configs/classifiers/`: Classifier-specific configurations
- `configs/models/`: Model-specific configurations
- `configs/dataset/`: Dataset configurations
- `configs/evaluation/`: Evaluation settings
- `configs/visualization/`: Visualization settings

## Error Handling

The modular architecture includes comprehensive error handling:

- **Model Loading**: Fallback strategies for different model sizes
- **Classifier Loading**: CPU fallback when GPU fails
- **Generation**: Memory management and batch size adjustment
- **Dataset Loading**: Caching and fallback datasets
- **Pipeline Execution**: Graceful failure with detailed error reporting

## Performance Considerations

- **Memory Management**: Automatic GPU memory cleanup
- **Caching**: Dataset caching for faster subsequent runs
- **Parallel Processing**: Optional parallel classifier evaluation
- **Batch Processing**: Configurable batch sizes for generation
- **Progress Tracking**: Detailed progress reporting for long operations

## Future Extensions

The modular architecture makes it easy to extend:

1. **New Models**: Add to configuration and ModelLoader
2. **New Classifiers**: Add to configuration and ClassifierManager
3. **New Datasets**: Extend DatasetManager
4. **New Visualizations**: Extend VisualizationManager
5. **New Analysis**: Create new evaluation components

## Support

For issues or questions about the modular architecture:

1. Check the test file for usage examples
2. Review the configuration files for settings
3. Examine the individual component documentation
4. Run the test suite to verify functionality 