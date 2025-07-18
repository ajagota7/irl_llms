#!/usr/bin/env python3
"""
ENHANCED WandB Analysis for Pythia-70m evaluation results with full Toxic-BERT category analysis.
This version handles all Toxic-BERT categories (toxic, severe_toxic, obscene, threat, insult, identity_hate)
and provides comprehensive visualization of how different types of toxicity change during detoxification.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wandb
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ========================================================================================
# CONFIGURATION
# ========================================================================================

WANDB_PROJECT = "pythia-detoxification-analysis"
WANDB_ENTITY = None

# Model colors
MODEL_COLORS = {
    'base': '#34495e',
    'detox_epoch_20': '#3498db',
    'detox_epoch_40': '#2ecc71', 
    'detox_epoch_60': '#f39c12',
    'detox_epoch_80': '#e74c3c',
    'detox_epoch_100': '#9b59b6'
}

# Classifier colors
CLASSIFIER_COLORS = {
    'toxic_bert': '#e74c3c',
    'roberta_toxicity': '#3498db',
    'dynabench_hate': '#2ecc71'
}

# Toxic-BERT category colors
TOXIC_BERT_CATEGORIES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
CATEGORY_COLORS = {
    'toxic': '#e74c3c',
    'severe_toxic': '#c0392b',
    'obscene': '#f39c12',
    'threat': '#e67e22',
    'insult': '#9b59b6',
    'identity_hate': '#8e44ad'
}

# ========================================================================================
# CORE FUNCTIONS
# ========================================================================================

def detect_models_and_classifiers(df: pd.DataFrame):
    """Detect models and classifiers from your actual column structure."""
    models = set()
    classifiers = set()
    toxic_bert_categories = set()
    
    # Known classifiers in your data
    known_classifiers = ['toxic_bert', 'roberta_toxicity', 'dynabench_hate']
    
    # Find all score columns
    score_cols = [col for col in df.columns if col.endswith('_score')]
    
    for col in score_cols:
        if col.startswith('prompt_'):
            # Format: prompt_{classifier}_score
            classifier = col.replace('prompt_', '').replace('_score', '')
            classifiers.add(classifier)
            
        elif col.startswith('full_'):
            # Format: full_{model}_{classifier}_score
            remainder = col.replace('full_', '').replace('_score', '')
            
            # Try to match known classifiers
            for known_classifier in known_classifiers:
                if remainder.endswith(known_classifier):
                    model = remainder.replace(f'_{known_classifier}', '')
                    models.add(model)
                    classifiers.add(known_classifier)
                    break
                    
        else:
            # Format: {model}_{classifier}_score (for output scores)
            for known_classifier in known_classifiers:
                if col.endswith(f'{known_classifier}_score'):
                    model = col.replace(f'_{known_classifier}_score', '')
                    models.add(model)
                    classifiers.add(known_classifier)
                    break
    
    # Check for Toxic-BERT category columns
    for col in score_cols:
        for category in TOXIC_BERT_CATEGORIES:
            if category in col and 'toxic_bert' in col:
                toxic_bert_categories.add(category)
    
    return sorted(list(models)), sorted(list(classifiers)), sorted(list(toxic_bert_categories))

def calculate_deltas(df: pd.DataFrame, models: List[str], classifiers: List[str], 
                    toxic_bert_categories: List[str]) -> pd.DataFrame:
    """Calculate delta scores between base model and other models."""
    print("📊 Calculating improvement deltas...")
    
    df_copy = df.copy()
    
    # Calculate deltas for main classifiers
    for classifier in classifiers:
        base_col = f'base_{classifier}_score'
        
        if base_col not in df_copy.columns:
            print(f"⚠️ Base column {base_col} not found")
            continue
            
        for model in models:
            if model == 'base':
                continue
                
            model_col = f'{model}_{classifier}_score'
            
            if model_col in df_copy.columns:
                delta_col = f'delta_{model}_vs_base_{classifier}'
                df_copy[delta_col] = df_copy[base_col] - df_copy[model_col]  # Positive = improvement
                print(f"✅ Created {delta_col}")
            else:
                print(f"⚠️ Model column {model_col} not found")
    
    # Calculate deltas for Toxic-BERT categories
    for category in toxic_bert_categories:
        base_col = f'base_toxic_bert_{category}_score'
        
        if base_col not in df_copy.columns:
            print(f"⚠️ Base Toxic-BERT category column {base_col} not found")
            continue
            
        for model in models:
            if model == 'base':
                continue
                
            model_col = f'{model}_toxic_bert_{category}_score'
            
            if model_col in df_copy.columns:
                delta_col = f'delta_{model}_vs_base_toxic_bert_{category}'
                df_copy[delta_col] = df_copy[base_col] - df_copy[model_col]  # Positive = improvement
                print(f"✅ Created {delta_col}")
            else:
                print(f"⚠️ Model Toxic-BERT category column {model_col} not found")
    
    return df_copy

def initialize_wandb(project_name: str = WANDB_PROJECT, entity: str = WANDB_ENTITY, 
                    config: Dict = None, wandb_key: str = None):
    """Initialize WandB with proper configuration."""
    
    if wandb_key and wandb_key != "your_key_here":
        try:
            wandb.login(key=wandb_key)
        except Exception as e:
            print(f"⚠️ WandB login failed: {e}")
    
    # Close any existing run
    if wandb.run is not None:
        wandb.finish()
    
    # Initialize new run
    try:
        run = wandb.init(
            project=project_name,
            entity=entity,
            name=f"pythia-detox-enhanced-{wandb.util.generate_id()}",
            config=config or {},
            reinit=True
        )
        
        print(f"✅ WandB initialized: {run.get_url()}")
        return run
    except Exception as e:
        print(f"❌ WandB initialization failed: {e}")
        return None

def log_plot_to_wandb(fig, plot_name: str, description: str = ""):
    """Log a plot to WandB."""
    if wandb.run is not None:
        wandb.log({plot_name: wandb.Plotly(fig)})
    else:
        print(f"📈 Created plot: {plot_name}")

# ========================================================================================
# VISUALIZATION FUNCTIONS
# ========================================================================================

def create_scatter_plots(df: pd.DataFrame, models: List[str], classifiers: List[str]):
    """Create interactive scatter plots for main classifiers."""
    print("\n📈 Creating scatter plots for main classifiers...")
    
    comparison_models = [m for m in models if m != 'base']
    
    for classifier in classifiers:
        fig = go.Figure()
        
        base_col = f'base_{classifier}_score'
        
        if base_col not in df.columns:
            print(f"⚠️ Skipping {classifier} - base column not found")
            continue
        
        for model in comparison_models:
            model_col = f'{model}_{classifier}_score'
            
            if model_col in df.columns:
                # Remove NaN values
                mask = ~(df[base_col].isna() | df[model_col].isna())
                x_data = df[mask][base_col]
                y_data = df[mask][model_col]
                
                if len(x_data) > 0:
                    improvement = (x_data - y_data).mean()
                    model_color = MODEL_COLORS.get(model, '#7f8c8d')
                    
                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode='markers',
                        name=f'{model.replace("detox_", "")} (Δ={improvement:.4f})',
                        marker=dict(color=model_color, size=4, opacity=0.6),
                        hovertemplate=f'<b>{model}</b><br>Base: %{{x:.4f}}<br>{model}: %{{y:.4f}}<extra></extra>',
                        visible=True if 'epoch_100' in model else 'legendonly'
                    ))
        
        # Add diagonal line
        if len(df[base_col].dropna()) > 0:
            score_cols = [col for col in df.columns if col.endswith(f'{classifier}_score')]
            all_scores = pd.concat([df[col].dropna() for col in score_cols if col in df.columns])
            if len(all_scores) > 0:
                max_val = all_scores.max()
                min_val = all_scores.min()
                
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines', name='No Change',
                    line=dict(color='red', dash='dash', width=2),
                    hoverinfo='none', showlegend=True
                ))
                
                fig.update_layout(
                    title=f'{classifier.replace("_", " ").title()}: Base vs Fine-tuned Models',
                    xaxis_title='Base Model Toxicity Score',
                    yaxis_title='Fine-tuned Model Toxicity Score',
                    height=600, showlegend=True,
                    xaxis=dict(range=[min_val, max_val]),
                    yaxis=dict(range=[min_val, max_val])
                )
                
                log_plot_to_wandb(fig, f"scatter_{classifier}", f"Scatter plot for {classifier}")

def create_toxic_bert_category_plots(df: pd.DataFrame, models: List[str], toxic_bert_categories: List[str]):
    """Create comprehensive Toxic-BERT category analysis plots."""
    print("\n🔍 Creating Toxic-BERT category analysis plots...")
    
    comparison_models = [m for m in models if m != 'base']
    
    # 1. Scatter plots for each category
    for category in toxic_bert_categories:
        fig = go.Figure()
        
        base_col = f'base_toxic_bert_{category}_score'
        
        if base_col not in df.columns:
            print(f"⚠️ Skipping {category} - base column not found")
            continue
        
        for model in comparison_models:
            model_col = f'{model}_toxic_bert_{category}_score'
            
            if model_col in df.columns:
                # Remove NaN values
                mask = ~(df[base_col].isna() | df[model_col].isna())
                x_data = df[mask][base_col]
                y_data = df[mask][model_col]
                
                if len(x_data) > 0:
                    improvement = (x_data - y_data).mean()
                    model_color = MODEL_COLORS.get(model, '#7f8c8d')
                    category_color = CATEGORY_COLORS.get(category, '#95a5a6')
                    
                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode='markers',
                        name=f'{model.replace("detox_", "")} (Δ={improvement:.4f})',
                        marker=dict(color=model_color, size=4, opacity=0.6),
                        hovertemplate=f'<b>{model} - {category}</b><br>Base: %{{x:.4f}}<br>{model}: %{{y:.4f}}<extra></extra>',
                        visible=True if 'epoch_100' in model else 'legendonly'
                    ))
        
        # Add diagonal line
        if len(df[base_col].dropna()) > 0:
            score_cols = [col for col in df.columns if col.endswith(f'{category}_score')]
            all_scores = pd.concat([df[col].dropna() for col in score_cols if col in df.columns])
            if len(all_scores) > 0:
                max_val = all_scores.max()
                min_val = all_scores.min()
                
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines', name='No Change',
                    line=dict(color='red', dash='dash', width=2),
                    hoverinfo='none', showlegend=True
                ))
                
                fig.update_layout(
                    title=f'Toxic-BERT {category.title()}: Base vs Fine-tuned Models',
                    xaxis_title='Base Model Score',
                    yaxis_title='Fine-tuned Model Score',
                    height=600, showlegend=True,
                    xaxis=dict(range=[min_val, max_val]),
                    yaxis=dict(range=[min_val, max_val])
                )
                
                log_plot_to_wandb(fig, f"scatter_toxic_bert_{category}", f"Scatter plot for Toxic-BERT {category}")
    
    # 2. Category comparison heatmap
    create_toxic_bert_heatmap(df, models, toxic_bert_categories)
    
    # 3. Category progression plots
    create_toxic_bert_progression(df, models, toxic_bert_categories)

def create_toxic_bert_heatmap(df: pd.DataFrame, models: List[str], toxic_bert_categories: List[str]):
    """Create heatmap comparing Toxic-BERT categories across models."""
    print("🏆 Creating Toxic-BERT category heatmap...")
    
    comparison_models = [m for m in models if m != 'base']
    comparison_data = []
    
    for model in comparison_models:
        row_data = {'model': model.replace('detox_', '')}
        
        for category in toxic_bert_categories:
            delta_col = f'delta_{model}_vs_base_toxic_bert_{category}'
            if delta_col in df.columns:
                avg_improvement = df[delta_col].mean()
                row_data[category] = avg_improvement
            else:
                row_data[category] = 0.0
        
        comparison_data.append(row_data)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create heatmap
    z_data = comparison_df[toxic_bert_categories].values
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=[c.title() for c in toxic_bert_categories],
        y=comparison_df['model'],
        colorscale='RdYlGn',
        colorbar=dict(title="Improvement"),
        hovertemplate='Model: %{y}<br>Category: %{x}<br>Improvement: %{z:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Toxic-BERT Categories: Model Performance Heatmap',
        height=400
    )
    
    log_plot_to_wandb(fig, "heatmap_toxic_bert_categories", "Toxic-BERT category comparison heatmap")

def create_toxic_bert_progression(df: pd.DataFrame, models: List[str], toxic_bert_categories: List[str]):
    """Create progression plots for Toxic-BERT categories."""
    print("📈 Creating Toxic-BERT category progression plots...")
    
    # Extract epoch information
    epoch_models = []
    for model in models:
        if model == 'base':
            epoch_models.append((0, model))
        elif 'epoch_' in model:
            try:
                epoch = int(model.split('epoch_')[1])
                epoch_models.append((epoch, model))
            except:
                continue
    
    epoch_models.sort(key=lambda x: x[0])
    epochs = [epoch for epoch, model in epoch_models]
    
    # Create subplot for all categories
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[c.title() for c in toxic_bert_categories],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    for i, category in enumerate(toxic_bert_categories):
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        improvements = []
        
        for epoch, model in epoch_models:
            if model == 'base':
                improvements.append(0.0)
            else:
                delta_col = f'delta_{model}_vs_base_toxic_bert_{category}'
                if delta_col in df.columns:
                    avg_improvement = df[delta_col].mean()
                    improvements.append(avg_improvement)
                else:
                    improvements.append(0)
        
        category_color = CATEGORY_COLORS.get(category, '#3498db')
        
        fig.add_trace(
            go.Scatter(
                x=epochs, y=improvements,
                mode='lines+markers',
                name=category.title(),
                line=dict(color=category_color, width=2),
                marker=dict(size=6),
                hovertemplate=f'<b>{category}</b><br>Epoch: %{{x}}<br>Avg Improvement: %{{y:.4f}}<extra></extra>',
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=1, row=row, col=col)
    
    fig.update_layout(
        title='Toxic-BERT Categories: Training Progression',
        height=800,
        showlegend=False
    )
    
    # Update axes labels
    for i in range(1, 3):
        for j in range(1, 4):
            fig.update_xaxes(title_text="Epoch", row=i, col=j)
            fig.update_yaxes(title_text="Improvement", row=i, col=j)
    
    log_plot_to_wandb(fig, "progression_toxic_bert_categories", "Toxic-BERT category progression")

def create_progression_plots(df: pd.DataFrame, models: List[str], classifiers: List[str]):
    """Create training progression plots for main classifiers."""
    print("\n📈 Creating progression plots for main classifiers...")
    
    # Extract epoch information
    epoch_models = []
    for model in models:
        if model == 'base':
            epoch_models.append((0, model))
        elif 'epoch_' in model:
            try:
                epoch = int(model.split('epoch_')[1])
                epoch_models.append((epoch, model))
            except:
                continue
    
    epoch_models.sort(key=lambda x: x[0])
    epochs = [epoch for epoch, model in epoch_models]
    
    # Combined progression plot
    fig = go.Figure()
    
    for classifier in classifiers:
        improvements = []
        
        for epoch, model in epoch_models:
            if model == 'base':
                improvements.append(0.0)
            else:
                delta_col = f'delta_{model}_vs_base_{classifier}'
                if delta_col in df.columns:
                    avg_improvement = df[delta_col].mean()
                    improvements.append(avg_improvement)
                else:
                    improvements.append(0)
        
        classifier_color = CLASSIFIER_COLORS.get(classifier, '#3498db')
        
        fig.add_trace(go.Scatter(
            x=epochs, y=improvements,
            mode='lines+markers',
            name=classifier.replace('_', ' ').title(),
            line=dict(color=classifier_color, width=3),
            marker=dict(size=8),
            hovertemplate=f'<b>{classifier}</b><br>Epoch: %{{x}}<br>Avg Improvement: %{{y:.4f}}<extra></extra>'
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)
    
    fig.update_layout(
        title='Training Progression: Average Improvement Over Epochs',
        xaxis_title='Training Epoch',
        yaxis_title='Average Toxicity Reduction',
        height=600, showlegend=True
    )
    
    log_plot_to_wandb(fig, "progression_all_classifiers", "Training progression across all classifiers")

def create_delta_plots(df: pd.DataFrame, models: List[str], classifiers: List[str]):
    """Create improvement distribution plots for main classifiers."""
    print("\n📊 Creating improvement distribution plots for main classifiers...")
    
    comparison_models = [m for m in models if m != 'base']
    
    for classifier in classifiers:
        fig = go.Figure()
        
        for model in comparison_models:
            delta_col = f'delta_{model}_vs_base_{classifier}'
            
            if delta_col in df.columns:
                delta_data = df[delta_col].dropna()
                
                if len(delta_data) > 0:
                    mean_improvement = delta_data.mean()
                    model_color = MODEL_COLORS.get(model, '#7f8c8d')
                    
                    fig.add_trace(go.Histogram(
                        x=delta_data,
                        name=f'{model.replace("detox_", "")} (μ={mean_improvement:.4f})',
                        opacity=0.7, marker_color=model_color, nbinsx=30,
                        hovertemplate=f'<b>{model}</b><br>Improvement: %{{x:.4f}}<br>Count: %{{y}}<extra></extra>'
                    ))
        
        fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=2)
        
        fig.update_layout(
            title=f'{classifier.replace("_", " ").title()}: Improvement Distribution',
            xaxis_title='Toxicity Reduction (positive = better)',
            yaxis_title='Frequency', barmode='overlay',
            height=600, showlegend=True
        )
        
        log_plot_to_wandb(fig, f"delta_{classifier}", f"Improvement distribution for {classifier}")

def create_toxic_bert_delta_plots(df: pd.DataFrame, models: List[str], toxic_bert_categories: List[str]):
    """Create improvement distribution plots for Toxic-BERT categories."""
    print("\n📊 Creating improvement distribution plots for Toxic-BERT categories...")
    
    comparison_models = [m for m in models if m != 'base']
    
    for category in toxic_bert_categories:
        fig = go.Figure()
        
        for model in comparison_models:
            delta_col = f'delta_{model}_vs_base_toxic_bert_{category}'
            
            if delta_col in df.columns:
                delta_data = df[delta_col].dropna()
                
                if len(delta_data) > 0:
                    mean_improvement = delta_data.mean()
                    model_color = MODEL_COLORS.get(model, '#7f8c8d')
                    
                    fig.add_trace(go.Histogram(
                        x=delta_data,
                        name=f'{model.replace("detox_", "")} (μ={mean_improvement:.4f})',
                        opacity=0.7, marker_color=model_color, nbinsx=30,
                        hovertemplate=f'<b>{model} - {category}</b><br>Improvement: %{{x:.4f}}<br>Count: %{{y}}<extra></extra>'
                    ))
        
        fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=2)
        
        fig.update_layout(
            title=f'Toxic-BERT {category.title()}: Improvement Distribution',
            xaxis_title='Toxicity Reduction (positive = better)',
            yaxis_title='Frequency', barmode='overlay',
            height=600, showlegend=True
        )
        
        log_plot_to_wandb(fig, f"delta_toxic_bert_{category}", f"Improvement distribution for Toxic-BERT {category}")

def create_heatmap(df: pd.DataFrame, models: List[str], classifiers: List[str]):
    """Create model comparison heatmap for main classifiers."""
    print("\n🏆 Creating comparison heatmap for main classifiers...")
    
    comparison_models = [m for m in models if m != 'base']
    comparison_data = []
    
    for model in comparison_models:
        row_data = {'model': model.replace('detox_', '')}
        
        for classifier in classifiers:
            delta_col = f'delta_{model}_vs_base_{classifier}'
            if delta_col in df.columns:
                avg_improvement = df[delta_col].mean()
                row_data[classifier] = avg_improvement
            else:
                row_data[classifier] = 0.0
        
        comparison_data.append(row_data)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create heatmap
    z_data = comparison_df[classifiers].values
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=[c.replace('_', ' ').title() for c in classifiers],
        y=comparison_df['model'],
        colorscale='RdYlGn',
        colorbar=dict(title="Improvement"),
        hovertemplate='Model: %{y}<br>Classifier: %{x}<br>Improvement: %{z:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Model Performance Heatmap: Improvement Across All Classifiers',
        height=400
    )
    
    log_plot_to_wandb(fig, "heatmap_comparison", "Model performance comparison heatmap")

def log_summary_metrics(df: pd.DataFrame, models: List[str], classifiers: List[str], 
                       toxic_bert_categories: List[str]):
    """Log comprehensive summary metrics to WandB."""
    print("\n📈 Logging comprehensive summary metrics...")
    
    if wandb.run is None:
        return
    
    metrics = {}
    comparison_models = [m for m in models if m != 'base']
    
    # Overall metrics for main classifiers
    for model in comparison_models:
        all_improvements = []
        
        for classifier in classifiers:
            delta_col = f'delta_{model}_vs_base_{classifier}'
            if delta_col in df.columns:
                all_improvements.extend(df[delta_col].dropna().tolist())
        
        if all_improvements:
            metrics[f"{model}_mean_improvement"] = np.mean(all_improvements)
            metrics[f"{model}_positive_rate"] = np.mean([x > 0.01 for x in all_improvements])
    
    # Classifier-specific metrics
    for classifier in classifiers:
        classifier_improvements = []
        
        for model in comparison_models:
            delta_col = f'delta_{model}_vs_base_{classifier}'
            if delta_col in df.columns:
                classifier_improvements.extend(df[delta_col].dropna().tolist())
        
        if classifier_improvements:
            metrics[f"{classifier}_overall_mean"] = np.mean(classifier_improvements)
    
    # Toxic-BERT category metrics
    for category in toxic_bert_categories:
        category_improvements = []
        
        for model in comparison_models:
            delta_col = f'delta_{model}_vs_base_toxic_bert_{category}'
            if delta_col in df.columns:
                category_improvements.extend(df[delta_col].dropna().tolist())
        
        if category_improvements:
            metrics[f"toxic_bert_{category}_overall_mean"] = np.mean(category_improvements)
            metrics[f"toxic_bert_{category}_positive_rate"] = np.mean([x > 0.01 for x in category_improvements])
    
    # Best model
    if comparison_models:
        best_model = max(comparison_models, 
            key=lambda m: metrics.get(f"{m}_mean_improvement", -1))
        metrics["best_overall_model"] = best_model
    
    # Best Toxic-BERT category
    if toxic_bert_categories:
        best_category = max(toxic_bert_categories,
            key=lambda c: metrics.get(f"toxic_bert_{c}_overall_mean", -1))
        metrics["best_toxic_bert_category"] = best_category
    
    wandb.log(metrics)

# ========================================================================================
# MAIN EXECUTION FUNCTION
# ========================================================================================

def run_complete_enhanced_analysis(csv_path: str, wandb_key: str = None):
    """Run complete enhanced WandB analysis with full Toxic-BERT category analysis."""
    
    print("🚀 STARTING ENHANCED WANDB ANALYSIS WITH TOXIC-BERT CATEGORIES")
    print("="*70)
    
    # Load and prepare data
    print("📊 Loading data...")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # Detect models, classifiers, and Toxic-BERT categories
    models, classifiers, toxic_bert_categories = detect_models_and_classifiers(df)
    print(f"📋 Detected models: {models}")
    print(f"🔍 Detected classifiers: {classifiers}")
    print(f"🎯 Detected Toxic-BERT categories: {toxic_bert_categories}")
    
    if not models or not classifiers:
        print("❌ No models or classifiers detected!")
        return None, None
    
    # Calculate deltas
    df = calculate_deltas(df, models, classifiers, toxic_bert_categories)
    
    # Initialize WandB
    config = {
        "dataset_size": len(df),
        "models": models,
        "classifiers": classifiers,
        "toxic_bert_categories": toxic_bert_categories
    }
    
    run = initialize_wandb(WANDB_PROJECT, config=config, wandb_key=wandb_key)
    
    # Create comprehensive visualizations
    try:
        # Main classifier visualizations
        create_scatter_plots(df, models, classifiers)
        create_progression_plots(df, models, classifiers)
        create_delta_plots(df, models, classifiers)
        create_heatmap(df, models, classifiers)
        
        # Toxic-BERT category visualizations
        if toxic_bert_categories:
            create_toxic_bert_category_plots(df, models, toxic_bert_categories)
            create_toxic_bert_delta_plots(df, models, toxic_bert_categories)
        
        # Log comprehensive metrics
        log_summary_metrics(df, models, classifiers, toxic_bert_categories)
        
        print("\n✅ All enhanced visualizations created successfully!")
        if run:
            print(f"🔗 View enhanced dashboard: {run.get_url()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    return df, run

# ========================================================================================
# USAGE
# ========================================================================================

if __name__ == "__main__":
    # Update with your actual file path
    csv_path = "/content/irl_llms/src/enhanced_toxicity_evaluation/results/pythia_70m_detox_comprehensive/toxicity_evaluation_results.csv"
    
    # Run enhanced analysis
    df, run = run_complete_enhanced_analysis(csv_path, wandb_key=None) 