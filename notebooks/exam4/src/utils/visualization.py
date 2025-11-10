"""
Visualization module.

This module provides functions to visualize training results, metrics,
and model performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Optional, List
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class Visualizer:
    """
    Class for creating visualizations of model performance.

    Provides methods to plot confusion matrices, ROC curves,
    learning curves, and other metrics.
    """

    def __init__(self, config: dict):
        """
        Initialize Visualizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.target_categories = config['data']['target_categories']

        logger.info("Visualizer initialized")

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: Optional[list] = None,
        normalize: bool = False,
        title: str = 'Confusion Matrix',
        output_path: Optional[str] = None,
        figsize: tuple = (10, 8)
    ):
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            classes: List of class names
            normalize: Whether to normalize the matrix
            title: Plot title
            output_path: Path to save the plot
            figsize: Figure size
        """
        if classes is None:
            classes = self.target_categories

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'

        # Create plot
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )

        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {output_path}")

        plt.show()
        plt.close()

    def plot_roc_curves(
        self,
        roc_metrics: Dict,
        title: str = 'ROC Curves',
        output_path: Optional[str] = None,
        figsize: tuple = (12, 8)
    ):
        """
        Plot ROC curves for multi-class classification.

        Args:
            roc_metrics: Dictionary with ROC metrics from MetricsCalculator
            title: Plot title
            output_path: Path to save the plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)

        colors = plt.cm.get_cmap('tab10')

        # Plot ROC curve for each class
        for i, class_name in enumerate(roc_metrics['classes']):
            fpr = roc_metrics['fpr'][class_name]
            tpr = roc_metrics['tpr'][class_name]
            auc_score = roc_metrics['auc'][class_name]

            plt.plot(
                fpr,
                tpr,
                color=colors(i),
                lw=2,
                label=f'{class_name} (AUC = {auc_score:.3f})'
            )

        # Plot micro-average if available
        if 'micro' in roc_metrics['auc']:
            plt.plot(
                roc_metrics['fpr']['micro'],
                roc_metrics['tpr']['micro'],
                color='deeppink',
                linestyle='--',
                lw=2,
                label=f'Micro-average (AUC = {roc_metrics["auc"]["micro"]:.3f})'
            )

        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {output_path}")

        plt.show()
        plt.close()

    def plot_precision_recall_curves(
        self,
        pr_metrics: Dict,
        title: str = 'Precision-Recall Curves',
        output_path: Optional[str] = None,
        figsize: tuple = (12, 8)
    ):
        """
        Plot Precision-Recall curves.

        Args:
            pr_metrics: Dictionary with PR metrics from MetricsCalculator
            title: Plot title
            output_path: Path to save the plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)

        colors = plt.cm.get_cmap('tab10')

        # Plot PR curve for each class
        for i, class_name in enumerate(pr_metrics['classes']):
            precision = pr_metrics['precision'][class_name]
            recall = pr_metrics['recall'][class_name]
            ap_score = pr_metrics['avg_precision'][class_name]

            plt.plot(
                recall,
                precision,
                color=colors(i),
                lw=2,
                label=f'{class_name} (AP = {ap_score:.3f})'
            )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curves saved to {output_path}")

        plt.show()
        plt.close()

    def plot_learning_curves(
        self,
        history: dict,
        metrics: Optional[List[str]] = None,
        title: str = 'Learning Curves',
        output_path: Optional[str] = None,
        figsize: tuple = (15, 5)
    ):
        """
        Plot training and validation learning curves.

        Args:
            history: Training history dictionary
            metrics: List of metrics to plot (if None, plots all)
            title: Plot title
            output_path: Path to save the plot
            figsize: Figure size
        """
        if metrics is None:
            # Get all metrics except validation metrics
            metrics = [k for k in history.keys() if not k.startswith('val_')]

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            ax = axes[i]

            # Plot training metric
            ax.plot(
                history[metric],
                label=f'Training {metric}',
                linewidth=2
            )

            # Plot validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in history:
                ax.plot(
                    history[val_metric],
                    label=f'Validation {metric}',
                    linewidth=2
                )

            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(metric.capitalize(), fontsize=11)
            ax.set_title(f'{metric.capitalize()} over Epochs', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning curves saved to {output_path}")

        plt.show()
        plt.close()

    def plot_feature_importance(
        self,
        feature_importance: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_n: int = 20,
        title: str = 'Feature Importance',
        output_path: Optional[str] = None,
        figsize: tuple = (12, 8)
    ):
        """
        Plot feature importance for tree-based models.

        Args:
            feature_importance: Feature importance values
            feature_names: List of feature names
            top_n: Number of top features to display
            title: Plot title
            output_path: Path to save the plot
            figsize: Figure size
        """
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(feature_importance))]

        # Sort features by importance
        indices = np.argsort(feature_importance)[::-1][:top_n]

        plt.figure(figsize=figsize)
        plt.barh(
            range(len(indices)),
            feature_importance[indices],
            color='steelblue'
        )

        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {output_path}")

        plt.show()
        plt.close()

    def plot_class_distribution(
        self,
        labels: np.ndarray,
        title: str = 'Class Distribution',
        output_path: Optional[str] = None,
        figsize: tuple = (10, 6)
    ):
        """
        Plot class distribution.

        Args:
            labels: Array of labels
            title: Plot title
            output_path: Path to save the plot
            figsize: Figure size
        """
        from collections import Counter

        # Count occurrences
        class_counts = Counter(labels)

        # Create plot
        plt.figure(figsize=figsize)

        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        plt.bar(classes, counts, color='steelblue', alpha=0.8)

        # Add count labels on bars
        for i, (cls, count) in enumerate(zip(classes, counts)):
            plt.text(i, count + max(counts)*0.01, str(count),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Class distribution plot saved to {output_path}")

        plt.show()
        plt.close()

    def plot_model_comparison(
        self,
        comparison: Dict[str, Dict[str, float]],
        title: str = 'Model Comparison',
        output_path: Optional[str] = None,
        figsize: tuple = (12, 6)
    ):
        """
        Plot comparison of multiple models.

        Args:
            comparison: Dictionary of model metrics
            title: Plot title
            output_path: Path to save the plot
            figsize: Figure size
        """
        models = list(comparison.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']

        x = np.arange(len(models))
        width = 0.2

        fig, ax = plt.subplots(figsize=figsize)

        for i, metric in enumerate(metrics):
            values = [comparison[model][metric] for model in models]
            ax.bar(x + i * width, values, width, label=metric.capitalize())

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='best', fontsize=10)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {output_path}")

        plt.show()
        plt.close()

    def plot_spectrogram_samples(
        self,
        spectrograms: np.ndarray,
        labels: np.ndarray,
        n_samples: int = 8,
        title: str = 'Spectrogram Samples',
        output_path: Optional[str] = None,
        figsize: tuple = (15, 10)
    ):
        """
        Plot sample spectrograms.

        Args:
            spectrograms: Array of spectrograms
            labels: Array of labels
            n_samples: Number of samples to plot
            title: Plot title
            output_path: Path to save the plot
            figsize: Figure size
        """
        n_samples = min(n_samples, len(spectrograms))
        n_cols = 4
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for i in range(n_samples):
            ax = axes[i]

            # Plot spectrogram
            if len(spectrograms[i].shape) == 3:
                # RGB image
                ax.imshow(spectrograms[i])
            else:
                # Grayscale
                ax.imshow(spectrograms[i], cmap='viridis', aspect='auto')

            ax.set_title(f'Label: {labels[i]}', fontsize=10)
            ax.axis('off')

        # Hide unused subplots
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')

        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Spectrogram samples saved to {output_path}")

        plt.show()
        plt.close()

    def create_interactive_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: Optional[list] = None,
        title: str = 'Interactive Confusion Matrix',
        output_path: Optional[str] = None
    ):
        """
        Create interactive confusion matrix using Plotly.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            classes: List of class names
            title: Plot title
            output_path: Path to save the HTML file
        """
        if classes is None:
            classes = self.target_categories

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 14},
            colorbar=dict(title="Count")
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            font=dict(size=12),
            height=600,
            width=700
        )

        if output_path:
            fig.write_html(output_path)
            logger.info(f"Interactive confusion matrix saved to {output_path}")

        fig.show()

    def generate_all_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        history: Optional[dict] = None,
        feature_importance: Optional[np.ndarray] = None,
        model_name: str = "Model",
        output_dir: str = "results/plots"
    ):
        """
        Generate all visualization plots.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            history: Training history (optional)
            feature_importance: Feature importance (optional)
            model_name: Name of the model
            output_dir: Output directory for plots
        """
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"\nGenerating all plots for {model_name}...")

        # Confusion matrix
        self.plot_confusion_matrix(
            y_true, y_pred,
            title=f'{model_name} - Confusion Matrix',
            output_path=os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
        )

        # Learning curves (if history available)
        if history is not None:
            self.plot_learning_curves(
                history,
                title=f'{model_name} - Learning Curves',
                output_path=os.path.join(output_dir, f'{model_name}_learning_curves.png')
            )

        # Feature importance (if available)
        if feature_importance is not None:
            self.plot_feature_importance(
                feature_importance,
                title=f'{model_name} - Feature Importance',
                output_path=os.path.join(output_dir, f'{model_name}_feature_importance.png')
            )

        logger.info(f"All plots saved to {output_dir}")
