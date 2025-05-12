            # Extract key correlation metrics
            model_cols = [col for col in results["correlations"].columns if col.startswith('seed_')]
            if len(model_cols) > 1:
                model_corr = results["correlations"].loc[model_cols, model_cols]
                avg_model_corr = (model_corr.sum().sum() - len(model_cols)) / (len(model_cols) * (len(model_cols) - 1))
                summary["average_model_correlation"] = avg_model_corr
                
            if "ground_truth" in results["correlations"].columns:
                gt_corr = results["correlations"].loc["ground_truth", model_cols].mean()
                summary["average_ground_truth_correlation"] = gt_corr
                
            if "true_label" in results["correlations"].columns:
                label_corr = results["correlations"].loc["true_label", model_cols].mean()
                summary["average_label_correlation"] = label_corr
        
        # Add ensemble improvement metrics
        if "best_comparison" in results:
            best_individual = results["best_comparison"]["best_individual"]
            best_ensemble = results["best_comparison"]["best_ensemble"]
            
            # Calculate improvement percentages
            accuracy_improvement = (best_ensemble["accuracy"] - best_individual["accuracy"]) / best_individual["accuracy"] * 100
            f1_improvement = (best_ensemble["f1"] - best_individual["f1"]) / best_individual["f1"] * 100
            auc_improvement = (best_ensemble["auc_roc"] - best_individual["auc_roc"]) / best_individual["auc_roc"] * 100
            
            summary["ensemble_improvements"] = {
                "accuracy_improvement_percent": accuracy_improvement,
                "f1_improvement_percent": f1_improvement,
                "auc_improvement_percent": auc_improvement,
                "best_individual_seed": best_individual["seed"],
                "best_ensemble_method": best_ensemble["method"]
            }
            
        # Add error pattern metrics
        if "error_patterns" in results:
            summary["disagreement_rate"] = results["error_patterns"]["disagreement_rate"]
            summary["all_wrong_count"] = results["error_patterns"]["all_wrong_count"]
            summary["some_right_count"] = results["error_patterns"]["some_right_count"]
            
        # Save summary
        with open(os.path.join(self.output_dir, 'analysis_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
            
        print("\nAnalysis Summary:")
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, float):
                        print(f"  {subkey}: {subvalue:.4f}")
                    else:
                        print(f"  {subkey}: {subvalue}")
            elif isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
                
        return summary


def analyze_across_model_sizes(
    model_sizes: List[str],
    seeds: List[int],
    checkpoints: Dict[str, int],
    output_dir: str = "reward_model_analysis"
):
    """
    Run analysis across different model sizes and compare results.
    
    Args:
        model_sizes: List of model sizes to analyze
        seeds: List of seeds to use for each model size
        checkpoints: Dictionary mapping model sizes to checkpoint numbers
        output_dir: Directory to save analysis results
    """
    print(f"Analyzing across model sizes: {model_sizes}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run analysis for each model size
    summaries = {}
    
    for model_size in model_sizes:
        print(f"\n{'='*50}")
        print(f"Analyzing model size: {model_size}")
        print(f"{'='*50}\n")
        
        checkpoint = checkpoints.get(model_size, 30)  # Default to 30 if not specified
        
        analyzer = RewardModelEnsembleAnalyzer(
            model_size=model_size,
            seeds=seeds,
            checkpoint=checkpoint,
            output_dir=output_dir
        )
        
        summary = analyzer.run_full_analysis()
        summaries[model_size] = summary
    
    # Compare results across model sizes
    print("\nComparing results across model sizes...")
    
    # Extract key metrics for comparison
    comparison = {
        "model_sizes": model_sizes,
        "seeds": seeds,
        "metrics": {}
    }
    
    # Define metrics to compare
    metrics_to_compare = [
        "average_model_correlation",
        "average_ground_truth_correlation",
        "average_label_correlation",
        "disagreement_rate"
    ]
    
    # Add ensemble improvement metrics
    ensemble_metrics = [
        "accuracy_improvement_percent",
        "f1_improvement_percent",
        "auc_improvement_percent"
    ]
    
    # Extract metrics
    for metric in metrics_to_compare:
        comparison["metrics"][metric] = {
            size: summaries[size].get(metric, None) for size in model_sizes
        }
        
    # Extract ensemble improvement metrics
    for metric in ensemble_metrics:
        comparison["metrics"][metric] = {
            size: summaries[size].get("ensemble_improvements", {}).get(metric, None) 
            for size in model_sizes
        }
    
    # Save comparison
    with open(os.path.join(output_dir, 'cross_model_comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Plot key metrics across model sizes
    for metric in metrics_to_compare + ensemble_metrics:
        values = [comparison["metrics"][metric].get(size, None) for size in model_sizes]
        
        # Skip if any values are None
        if None in values:
            continue
            
        plt.figure(figsize=(10, 6))
        plt.bar(model_sizes, values)
        plt.xlabel('Model Size')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} Across Model Sizes')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{metric}.png'))
        plt.close()
    
    print("\nCross-model comparison complete!")
    return comparison


def weighted_ensemble_predictions(
    model_size: str,
    seeds: List[int],
    checkpoint: int,
    weights: Optional[Dict[int, float]] = None,
    texts: List[str] = None,
    dataset_path: Optional[str] = None
):
    """
    Generate predictions using a weighted ensemble of reward models.
    
    Args:
        model_size: Size of the model (70m, 160m, 410m, 1b)
        seeds: List of seeds to use in the ensemble
        checkpoint: Checkpoint number to use
        weights: Optional dictionary mapping seeds to weights (defaults to equal weights)
        texts: List of texts to generate predictions for
        dataset_path: Optional path to a dataset to use instead of texts
        
    Returns:
        Array of ensemble predictions
    """
    # Initialize analyzer to load models
    analyzer = RewardModelEnsembleAnalyzer(
        model_size=model_size,
        seeds=seeds,
        checkpoint=checkpoint,
        output_dir="temp"  # Temporary directory
    )
    
    # Use equal weights if not provided
    if weights is None:
        weights = {seed: 1.0 / len(seeds) for seed in seeds}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {seed: weight / total_weight for seed, weight in weights.items()}
    
    # Load texts from dataset if provided
    if texts is None and dataset_path is not None:
        try:
            ds = load_dataset(dataset_path)
            if isinstance(ds, dict) and 'train' in ds:
                ds = ds['train']
            
            # Extract texts from dataset
            if 'text' in ds.column_names:
                texts = ds['text']
            elif 'output' in ds.column_names:
                texts = ds['output']
            else:
                raise ValueError(f"Could not find text column in dataset: {dataset_path}")
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    if texts is None:
        raise ValueError("Either texts or dataset_path must be provided")
    
    # Get predictions from each model
    all_predictions = {}
    for seed, model in analyzer.reward_models.items():
        if seed in weights:
            predictions = analyzer.get_model_predictions(
                texts, 
                model, 
                analyzer.reward_tokenizers[seed]
            )
            all_predictions[seed] = predictions
    
    # Calculate weighted ensemble predictions
    ensemble_preds = np.zeros(len(texts))
    for seed, preds in all_predictions.items():
        ensemble_preds += weights[seed] * preds
    
    return ensemble_preds


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description="Analyze reward model ensembles")
    parser.add_argument("--model_sizes", nargs="+", default=["70m", "160m", "410m", "1b"],
                        help="Model sizes to analyze")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 200, 300, 400],
                        help="Seeds to analyze")
    parser.add_argument("--output_dir", default="reward_model_analysis",
                        help="Directory to save analysis results")
    parser.add_argument("--cross_model", action="store_true",
                        help="Run analysis across model sizes")
    
    args = parser.parse_args()
    
    # Define checkpoints for each model size
    checkpoints = {
        "70m": 30,
        "160m": 50,
        "410m": 70,
        "1b": 70
    }
    
    if args.cross_model:
        # Run analysis across model sizes
        analyze_across_model_sizes(
            model_sizes=args.model_sizes,
            seeds=args.seeds,
            checkpoints=checkpoints,
            output_dir=args.output_dir
        )
    else:
        # Run analysis for each model size separately
        for model_size in args.model_sizes:
            print(f"\n{'='*50}")
            print(f"Analyzing model size: {model_size}")
            print(f"{'='*50}\n")
            
            checkpoint = checkpoints.get(model_size, 30)  # Default to 30 if not specified
            
            analyzer = RewardModelEnsembleAnalyzer(
                model_size=model_size,
                seeds=args.seeds,
                checkpoint=checkpoint,
                output_dir=args.output_dir
            )
            
            analyzer.run_full_analysis()


if __name__ == "__main__":
    main() 