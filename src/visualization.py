"""
Visualization Module for Rainfall Prediction System.
Handles plotting of EDA charts, confusion matrices, ROC curves,
performance comparison bar charts, and feature importances.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """
    Plots and saves a single confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Rain (0)', 'Rain (1)'],
                yticklabels=['No Rain (0)', 'Rain (1)'],
                annot_kws={'size': 13, 'weight': 'bold'})
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=11)
    plt.ylabel('True Label', fontsize=11)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curves(roc_data_list, save_path=None):
    """
    Plots multiple ROC curves on a single axes.
    
    Parameters:
        roc_data_list (list): List of dicts or tuples containing:
                              (name, y_true, y_prob, color, linestyle, linewidth)
    """
    plt.figure(figsize=(7, 6))
    for item in roc_data_list:
        name = item['name']
        y_true = item['y_true']
        y_prob = item['y_prob']
        color = item.get('color', None)
        linestyle = item.get('linestyle', '-')
        linewidth = item.get('linewidth', 2)
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})",
                 color=color, linestyle=linestyle, linewidth=linewidth)
        
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance (AUC = 0.5000)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Recall)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_model_comparison(comparison_df, save_path=None):
    """
    Plots a multi-metric bar chart comparing Accuracy, Precision, Recall, and F1 across models.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    models = comparison_df['Model'].tolist()
    
    accs = comparison_df['Test Accuracy'].tolist()
    precs = comparison_df['Precision'].tolist()
    recs = comparison_df['Recall'].tolist()
    f1s = comparison_df['F1'].tolist()
    
    x = np.arange(len(models))
    width = 0.18
    
    ax.bar(x - 1.5 * width, accs, width, label='Accuracy', color='#3498db', alpha=0.9)
    ax.bar(x - 0.5 * width, precs, width, label='Precision', color='#9b59b6', alpha=0.9)
    ax.bar(x + 0.5 * width, recs, width, label='Recall', color='#e67e22', alpha=0.9)
    ax.bar(x + 1.5 * width, f1s, width, label='F1-Score', color='#2ecc71', alpha=0.9)
    
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Model Performance Comparison on Untouched Test Set', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    
    # Shorten labels for aesthetic display
    short_labels = [m.replace('Logistic Regression', 'LR').replace('Decision Tree', 'DT').replace('Random Forest', 'RF') for m in models]
    ax.set_xticklabels(short_labels, fontsize=10, fontweight='bold')
    ax.set_ylim([0.65, 1.02])
    ax.axhline(0.8, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_feature_importance(model, feature_names, save_path=None, csv_path=None):
    """
    Plots and exports feature importances from a tree-based model.
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError(f"Model {type(model).__name__} does not have feature_importances_ attribute.")
        
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    if csv_path:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        feat_imp_df.to_csv(csv_path, index=False)
        
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='Blues_r', hue='Feature', legend=False)
    plt.title('Random Forest (Tuned) — Feature Importances', fontsize=13, fontweight='bold')
    plt.xlabel('Gini Importance (Mean Decrease Impurity)', fontsize=11)
    plt.ylabel('Feature', fontsize=11)
    
    for p in ax.patches:
        w = p.get_width()
        ax.annotate(f'{w:.4f} ({w*100:.1f}%)',
                    (w, p.get_y() + p.get_height() / 2.),
                    xytext=(5, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=10)
                    
    plt.xlim(0, max(importances) * 1.25)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
    return feat_imp_df

def plot_eda_figures(raw_df, save_dir="outputs/figures"):
    """
    Generates and saves the core exploratory figures:
    - target_class_distribution.png
    - correlation_heatmap.png
    - rainfall_relationships.png
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Target Class Distribution
    plt.figure(figsize=(6, 4))
    raw_counts = raw_df['rainfall'].value_counts()
    ax = sns.barplot(x=raw_counts.index.astype(str), y=raw_counts.values,
                     palette=['#2ecc71', '#e74c3c'], hue=raw_counts.index.astype(str), legend=False)
    plt.title('Rainfall Target Class Distribution (Raw Dataset)', fontsize=12, fontweight='bold')
    plt.xlabel('Rainfall')
    plt.ylabel('Count')
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "target_class_distribution.png"), dpi=300)
    plt.close()
    
    # 2. Correlation Heatmap
    numeric_df = raw_df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Feature Correlation Heatmap', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "correlation_heatmap.png"), dpi=300)
    plt.close()
    
    # 3. Weather Relationships
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    rain_col = raw_df['rainfall']
    
    for idx, (var, pal, color_name) in enumerate([
        ('humidity', ['#95a5a6', '#3498db'], 'Humidity'),
        ('sunshine', ['#95a5a6', '#f39c12'], 'Sunshine'),
        ('cloud',    ['#95a5a6', '#34495e'], 'Cloud Cover')
    ]):
        if var in raw_df.columns:
            sns.boxplot(x=rain_col, y=raw_df[var], ax=axes[idx], palette=pal, hue=rain_col, legend=False)
            axes[idx].set_title(f'{color_name} vs Rainfall', fontweight='bold')
            axes[idx].set_xlabel('Rainfall')
            axes[idx].set_ylabel(var)
            
    plt.suptitle('Weather Variables by Rainfall Occurrence', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rainfall_relationships.png"), dpi=300)
    plt.close()
