#!/usr/bin/env python3
"""
Enhanced Toxicity Evaluation Pipeline - Usage Examples and Setup
"""

import os
import sys
from pathlib import Path
import argparse
import wandb
from omegaconf import OmegaConf

# Add the current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from enhanced_toxicity_pipeline import EnhancedToxicityEvaluator, ExperimentConfig, ModelConfig
from enhanced_inspector import EnhancedToxicityInspector, quick_multi_model_analysis


def setup_wandb():
    """Setup W&B with user credentials."""
    print("🔧 Setting up Weights & Biases...")
    
    # Check if already logged in
    try:
        wandb.api.viewer()
        print("✅ Already logged into W&B")
        return True
    except:
        print("❌ Not logged into W&B")
        
    # Prompt for login
    api_key = input("Enter your W&B API key (or press Enter to skip W&B logging): ").strip()
    
    if api_key:
        try:
            wandb.login(key=api_key)
            print("✅ Successfully logged into W&B")
            return True
        except Exception as e:
            print(f"❌ Failed to login to W&B: {e}")
            return False
    else:
        print("⚠️  Skipping W&B logging")
        return False


def create_example_configs():
    """Create example configuration files."""
    print("📁 Creating example configuration files...")
    
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    # Main config
    main_config = {
        "defaults": [
            "models: pythia_detox",
            "dataset: high_toxicity", 
            "classifiers: all_classifiers",
            "visualization: comprehensive",
            "_self_"
        ],
        "experiment_name": "toxicity_evaluation_${now:%Y%m%d_%H%M%S}",
        "output_dir": "results/${experiment_name}",
        "seed": 42,
        "device": "auto",
        "generation_params": {
            "max_new_tokens": 100,
            "do_sample": True,
            "temperature": 0.7,
            "use_cache": True
        },
        "wandb_project": "toxicity-evaluation",
        "wandb_entity": None,
        "wandb_tags": ["toxicity", "evaluation", "rlhf"]
    }
    
    with open(configs_dir / "config.yaml", "w") as f:
        OmegaConf.save(main_config, f)
    
    # Create subdirectories
    for subdir in ["models", "dataset", "classifiers", "visualization"]:
        (configs_dir / subdir).mkdir(exist_ok=True)
    
    # Models configurations
    models_configs = {
        "pythia_detox.yaml": {
            "models": [
                {
                    "name": "base",
                    "hf_path": "EleutherAI/pythia-410m",
                    "epoch": None,
                    "checkpoint_pattern": None
                },
                {
                    "name": "detox_epoch_50", 
                    "hf_path": "ajagota71/pythia-410m-s-nlp-detox",
                    "epoch": 50,
                    "checkpoint_pattern": "checkpoint-epoch-{epoch}"
                },
                {
                    "name": "detox_epoch_100",
                    "hf_path": "ajagota71/pythia-410m-s-nlp-detox", 
                    "epoch": 100,
                    "checkpoint_pattern": "checkpoint-epoch-{epoch}"
                }
            ]
        },
        "rlhf_progression.yaml": {
            "models": [
                {
                    "name": "base",
                    "hf_path": "EleutherAI/pythia-410m"
                },
                {
                    "name": "rlhf_step_1000",
                    "hf_path": "your-org/model-rlhf",
                    "epoch": 1000,
                    "checkpoint_pattern": "checkpoint-{epoch}"
                },
                {
                    "name": "rlhf_step_2000", 
                    "hf_path": "your-org/model-rlhf",
                    "epoch": 2000,
                    "checkpoint_pattern": "checkpoint-{epoch}"
                },
                {
                    "name": "rlhf_final",
                    "hf_path": "your-org/model-rlhf-final"
                }
            ]
        }
    }
    
    for filename, config in models_configs.items():
        with open(configs_dir / "models" / filename, "w") as f:
            OmegaConf.save(config, f)
    
    # Dataset configurations
    dataset_configs = {
        "high_toxicity.yaml": {
            "name": "allenai/real-toxicity-prompts",
            "split": "train", 
            "max_prompts": 5000,
            "shuffle_seed": 42,
            "filter_criteria": {
                "prompt_toxicity_min": 0.6,
                "prompt_toxicity_max": 1.0,
                "metrics": ["toxicity", "severe_toxicity"]
            }
        },
        "comprehensive.yaml": {
            "name": "allenai/real-toxicity-prompts",
            "split": "train",
            "max_prompts": 10000, 
            "shuffle_seed": 42,
            "filter_criteria": {
                "prompt_toxicity_min": 0.2,
                "prompt_toxicity_max": 1.0,
                "metrics": ["toxicity", "severe_toxicity", "profanity", "threat", "insult", "identity_attack"]
            }
        }
    }
    
    for filename, config in dataset_configs.items():
        with open(configs_dir / "dataset" / filename, "w") as f:
            OmegaConf.save(config, f)
    
    # Classifier configurations
    classifier_configs = {
        "all_classifiers.yaml": {
            "roberta_toxicity": True,
            "dynabench_hate": True,
            "unitary_toxic_bert": True,
            "perspective_api": False,
            "batch_size": 64,
            "max_length": 512
        },
        "lightweight.yaml": {
            "roberta_toxicity": True,
            "dynabench_hate": False,
            "unitary_toxic_bert": True,
            "perspective_api": False,
            "batch_size": 32,
            "max_length": 256
        }
    }
    
    for filename, config in classifier_configs.items():
        with open(configs_dir / "classifiers" / filename, "w") as f:
            OmegaConf.save(config, f)
    
    # Visualization configurations  
    viz_configs = {
        "comprehensive.yaml": {
            "plot_individual_classifiers": True,
            "plot_category_distributions": True,
            "create_scatter_plots": True,
            "analyze_prompt_only": True,
            "analyze_output_only": True,
            "analyze_prompt_plus_output": True,
            "save_interactive_plots": True,
            "wandb_log_plots": True
        },
        "basic.yaml": {
            "plot_individual_classifiers": True,
            "plot_category_distributions": False,
            "create_scatter_plots": True,
            "analyze_prompt_only": False,
            "analyze_output_only": True,
            "analyze_prompt_plus_output": True,
            "save_interactive_plots": False,
            "wandb_log_plots": True
        }
    }
    
    for filename, config in viz_configs.items():
        with open(configs_dir / "visualization" / filename, "w") as f:
            OmegaConf.save(config, f)
    
    print("✅ Created example configuration files")


def run_basic_example():
    """Run a basic example evaluation."""
    print("🚀 Running basic example evaluation...")
    
    # Create a simple configuration programmatically
    config = ExperimentConfig(
        experiment_name="basic_example",
        output_dir="results/basic_example",
        models=[
            ModelConfig(name="base", hf_path="EleutherAI/pythia-410m"),
            ModelConfig(
                name="detox", 
                hf_path="ajagota71/pythia-410m-s-nlp-detox-checkpoint-epoch-100"
            )
        ],
        wandb_tags=["example", "basic"]
    )
    
    # Limit dataset size for quick example
    config.dataset.max_prompts = 100
    
    try:
        # Run evaluation
        evaluator = EnhancedToxicityEvaluator(config)
        results_df = evaluator.run_evaluation()
        
        print("✅ Basic evaluation complete!")
        
        # Run quick analysis
        inspector = EnhancedToxicityInspector(results_df, wandb_logging=False)
        inspector.export_comprehensive_report(output_dir="results/basic_example/inspection")
        
        print("🎉 Example complete! Check results/basic_example/ for outputs")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        raise


def run_multi_checkpoint_example():
    """Run an example with multiple checkpoints."""
    print("🚀 Running multi-checkpoint example...")
    
    config = ExperimentConfig(
        experiment_name="multi_checkpoint_example",
        output_dir="results/multi_checkpoint_example", 
        models=[
            ModelConfig(name="base", hf_path="EleutherAI/pythia-410m"),
            ModelConfig(
                name="detox_epoch_50",
                hf_path="ajagota71/pythia-410m-s-nlp-detox",
                epoch=50,
                checkpoint_pattern="checkpoint-epoch-{epoch}"
            ),
            ModelConfig(
                name="detox_epoch_100", 
                hf_path="ajagota71/pythia-410m-s-nlp-detox",
                epoch=100,
                checkpoint_pattern="checkpoint-epoch-{epoch}"
            )
        ],
        wandb_tags=["multi-checkpoint", "progression"]
    )
    
    # Moderate dataset size
    config.dataset.max_prompts = 1000
    
    try:
        # Run evaluation
        evaluator = EnhancedToxicityEvaluator(config)
        results_df = evaluator.run_evaluation()
        
        print("✅ Multi-checkpoint evaluation complete!")
        
        # Advanced analysis
        inspector = EnhancedToxicityInspector(
            results_df, 
            model_names=["base", "detox_epoch_50", "detox_epoch_100"],
            wandb_logging=True
        )
        
        # Generate comprehensive analysis
        inspector.export_comprehensive_report(output_dir="results/multi_checkpoint_example/inspection")
        
        # Show progression plots
        if "unitary_toxic_bert" in inspector.classifiers:
            for category in ["toxicity", "severe_toxicity", "insult"]:
                inspector.create_model_progression_plot(category=category)
        
        # Quick rankings
        inspector.quick_model_ranking(category="toxicity")
        
        print("🎉 Multi-checkpoint analysis complete!")
        
    except Exception as e:
        print(f"❌ Multi-checkpoint example failed: {e}")
        raise


def analyze_existing_results(csv_path: str, model_names: List[str] = None):
    """Analyze existing results from a CSV file."""
    print(f"📊 Analyzing existing results from {csv_path}")
    
    if not Path(csv_path).exists():
        print(f"❌ File not found: {csv_path}")
        return
    
    try:
        # Quick analysis
        quick_multi_model_analysis(csv_path, model_names)
        
        # Detailed analysis
        inspector = EnhancedToxicityInspector(
            pd.read_csv(csv_path), 
            model_names=model_names,
            wandb_logging=False
        )
        
        # Interactive exploration
        print("\n🔍 Starting interactive prompt explorer...")
        inspector.interactive_prompt_explorer(n_samples=10)
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description="Enhanced Toxicity Evaluation Pipeline")
    parser.add_argument("--setup", action="store_true", help="Setup configuration files and W&B")
    parser.add_argument("--example", choices=["basic", "multi"], help="Run example evaluation")
    parser.add_argument("--analyze", type=str, help="Analyze existing CSV results")
    parser.add_argument("--models", nargs="+", help="Model names for analysis")
    parser.add_argument("--config", type=str, help="Path to Hydra config file")
    parser.add_argument("--wandb-entity", type=str, help="W&B entity name")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_wandb()
        create_example_configs()
        print("✅ Setup complete!")
        return
    
    if args.example:
        if args.example == "basic":
            run_basic_example()
        elif args.example == "multi":
            run_multi_checkpoint_example()
        return
    
    if args.analyze:
        analyze_existing_results(args.analyze, args.models)
        return
    
    if args.config:
        # Run with Hydra config
        print(f"🚀 Running evaluation with config: {args.config}")
        
        # Load config
        config = OmegaConf.load(args.config)
        
        # Override W&B entity if provided
        if args.wandb_entity:
            config.wandb_entity = args.wandb_entity
        
        # Convert to our config format
        experiment_config = ExperimentConfig(**config)
        
        # Run evaluation
        evaluator = EnhancedToxicityEvaluator(experiment_config)
        results_df = evaluator.run_evaluation()
        
        print("🎉 Evaluation complete!")
        return
    
    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()


# Additional utility functions for notebook usage
def notebook_quick_start():
    """Quick start function for Jupyter notebooks."""
    print("📓 NOTEBOOK QUICK START")
    print("=" * 50)
    
    print("""
    # Example 1: Basic Two-Model Comparison
    from enhanced_toxicity_pipeline import EnhancedToxicityEvaluator, ExperimentConfig, ModelConfig
    
    config = ExperimentConfig(
        experiment_name="notebook_example",
        models=[
            ModelConfig(name="base", hf_path="EleutherAI/pythia-410m"),
            ModelConfig(name="detox", hf_path="ajagota71/pythia-410m-s-nlp-detox-checkpoint-epoch-100")
        ]
    )
    config.dataset.max_prompts = 500  # Small for quick testing
    
    evaluator = EnhancedToxicityEvaluator(config)
    results_df = evaluator.run_evaluation()
    
    # Example 2: Analyze Results
    from enhanced_inspector import EnhancedToxicityInspector
    
    inspector = EnhancedToxicityInspector(results_df, wandb_logging=False)
    inspector.quick_model_ranking()
    inspector.create_toxicity_radar_chart()
    inspector.export_comprehensive_report()
    
    # Example 3: Load and Analyze Existing Results
    inspector = load_and_inspect_enhanced("results/comprehensive_results.csv")
    inspector.analyze_improvement_patterns()
    inspector.interactive_prompt_explorer()
    """)


def create_requirements_file():
    """Create requirements.txt file."""
    requirements = """
# Core dependencies
torch>=1.12.0
transformers>=4.21.0
datasets>=2.0.0
pandas>=1.5.0
numpy>=1.21.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.10.0

# ML/Stats
scipy>=1.9.0
scikit-learn>=1.1.0

# Configuration and logging
hydra-core>=1.2.0
omegaconf>=2.2.0
wandb>=0.13.0

# Development
jupyter>=1.0.0
ipywidgets>=7.7.0
tqdm>=4.64.0

# Optional for GPU optimization
# accelerate>=0.12.0
# bitsandbytes>=0.35.0
""".strip()
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("✅ Created requirements.txt")


def setup_project_structure():
    """Set up the complete project structure."""
    print("📁 Setting up project structure...")
    
    # Create directories
    directories = [
        "configs",
        "configs/models", 
        "configs/dataset",
        "configs/classifiers",
        "configs/visualization",
        "results",
        "data",
        "notebooks",
        "scripts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True, parents=True)
    
    # Create gitignore
    gitignore_content = """
# Data and results
results/
data/
*.csv
*.json
*.html

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv
pip-log.txt

# Jupyter
.ipynb_checkpoints
*.ipynb

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# W&B
wandb/

# Model files
*.bin
*.safetensors
*.pt
*.pth

# Logs
*.log
logs/
""".strip()
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    # Create README
    readme_content = """
# Enhanced Toxicity Evaluation Pipeline

A comprehensive pipeline for evaluating toxicity in language models with support for:
- Multiple models and checkpoints
- Multiple toxicity classifiers
- Comprehensive visualizations
- W&B integration
- Hydra configuration management

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Setup configuration and W&B:
```bash
python usage_example.py --setup
```

3. Run basic example:
```bash
python usage_example.py --example basic
```

## Usage

### Command Line
```bash
# Run with Hydra config
python usage_example.py --config configs/config.yaml

# Analyze existing results  
python usage_example.py --analyze results/comprehensive_results.csv

# Multi-checkpoint example
python usage_example.py --example multi
```

### Python API
```python
from enhanced_toxicity_pipeline import EnhancedToxicityEvaluator, ExperimentConfig, ModelConfig

config = ExperimentConfig(
    experiment_name="my_experiment",
    models=[
        ModelConfig(name="base", hf_path="EleutherAI/pythia-410m"),
        ModelConfig(name="finetuned", hf_path="your-model-path")
    ]
)

evaluator = EnhancedToxicityEvaluator(config)
results = evaluator.run_evaluation()
```

## Features

- **Multi-Model Support**: Compare base models with multiple checkpoints
- **Advanced Classifiers**: RoBERTa, DynaBench, Unitary Toxic-BERT
- **Rich Visualizations**: Radar charts, heatmaps, scatter plots, progression plots
- **Statistical Analysis**: Significance testing, improvement patterns
- **Interactive Exploration**: Prompt-by-prompt analysis
- **W&B Integration**: Automatic logging and artifact management

## Configuration

The pipeline uses Hydra for configuration management. See `configs/` directory for examples.

## Analysis Tools

The `EnhancedToxicityInspector` provides comprehensive analysis capabilities:
- Model rankings and comparisons
- Toxicity category breakdowns
- Prompt sensitivity analysis
- Interactive exploration tools
- Comprehensive HTML reports
""".strip()
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    
    # Create example notebook
    notebook_content = """
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Enhanced Toxicity Evaluation - Example Notebook\\n",
    "\\n",
    "This notebook demonstrates the Enhanced Toxicity Evaluation Pipeline."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import libraries\\n",
    "import sys\\n",
    "sys.path.append('..')\\n",
    "\\n",
    "from enhanced_toxicity_pipeline import EnhancedToxicityEvaluator, ExperimentConfig, ModelConfig\\n",
    "from enhanced_inspector import EnhancedToxicityInspector\\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code", 
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Configure experiment\\n",
    "config = ExperimentConfig(\\n",
    "    experiment_name=\\"notebook_experiment\\",\\n",
    "    models=[\\n",
    "        ModelConfig(name=\\"base\\", hf_path=\\"EleutherAI/pythia-410m\\"),\\n",
    "        ModelConfig(name=\\"detox\\", hf_path=\\"ajagota71/pythia-410m-s-nlp-detox-checkpoint-epoch-100\\")\\n",
    "    ]\\n",
    ")\\n",
    "\\n",
    "# Reduce dataset size for quick testing\\n",
    "config.dataset.max_prompts = 200"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run evaluation\\n",
    "evaluator = EnhancedToxicityEvaluator(config)\\n",
    "results_df = evaluator.run_evaluation()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Analyze results\\n",
    "inspector = EnhancedToxicityInspector(results_df, wandb_logging=False)\\n",
    "\\n",
    "# Quick summary\\n",
    "summary = inspector.get_model_comparison_summary()\\n",
    "display(summary)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create visualizations\\n",
    "radar_chart = inspector.create_toxicity_radar_chart()\\n",
    "radar_chart.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Improvement analysis\\n",
    "improvement_analysis = inspector.analyze_improvement_patterns()\\n",
    "print(f\\"Analysis for {len(improvement_analysis['comparison_models'])} models\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Export comprehensive report\\n",
    "inspector.export_comprehensive_report(output_dir=\\"notebook_results\\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
""".strip()
    
    with open("notebooks/example_evaluation.ipynb", "w") as f:
        f.write(notebook_content)
    
    create_requirements_file()
    create_example_configs()
    
    print("✅ Project structure created successfully!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Setup W&B: python usage_example.py --setup")
    print("3. Run example: python usage_example.py --example basic")


# If running this script directly, set up the project
if __name__ == "__main__" and len(sys.argv) == 1:
    print("🚀 Enhanced Toxicity Evaluation Pipeline Setup")
    print("=" * 60)
    
    choice = input("Setup complete project structure? (y/n): ").strip().lower()
    if choice == 'y':
        setup_project_structure()
    else:
        main()