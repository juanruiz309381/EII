"""
Metrics calculation module.

This module provides functions to calculate various classification metrics
and generate comprehensive evaluation reports.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)
from typing import Dict, Tuple, Optional
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Class for calculating classification metrics.

    Provides comprehensive metrics including accuracy, precision,
    recall, F1-score, and various curves (ROC, PR).
    """

    def __init__(self, config: dict):
        """
        Initialize MetricsCalculator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.metrics_config = config['metrics']
        self.target_categories = config['data']['target_categories']

        logger.info("MetricsCalculator initialized")

    def calculate_basic_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """
        Calculate basic classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging method for multi-class metrics

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # Precision
        metrics['precision'] = precision_score(
            y_true, y_pred, average=average, zero_division=0
        )

        # Recall
        metrics['recall'] = recall_score(
            y_true, y_pred, average=average, zero_division=0
        )

        # F1 Score
        metrics['f1_score'] = f1_score(
            y_true, y_pred, average=average, zero_division=0
        )

        logger.info(f"\nBasic Metrics ({average} average):")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name:15}: {value:.4f}")

        return metrics

    def calculate_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-class metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary of per-class metrics
        """
        # Calculate metrics for each class
        precision = precision_score(
            y_true, y_pred, average=None, zero_division=0
        )
        recall = recall_score(
            y_true, y_pred, average=None, zero_division=0
        )
        f1 = f1_score(
            y_true, y_pred, average=None, zero_division=0
        )

        # Get unique classes
        classes = np.unique(np.concatenate([y_true, y_pred]))

        per_class_metrics = {}

        logger.info("\nPer-class Metrics:")
        logger.info(f"{'Class':15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
        logger.info("-" * 50)

        for i, class_name in enumerate(classes):
            per_class_metrics[class_name] = {
                'precision': float(precision[i]) if i < len(precision) else 0.0,
                'recall': float(recall[i]) if i < len(recall) else 0.0,
                'f1_score': float(f1[i]) if i < len(f1) else 0.0
            }

            logger.info(
                f"{class_name:15} "
                f"{per_class_metrics[class_name]['precision']:10.4f} "
                f"{per_class_metrics[class_name]['recall']:10.4f} "
                f"{per_class_metrics[class_name]['f1_score']:10.4f}"
            )

        return per_class_metrics

    def calculate_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: Optional[str] = None
    ) -> np.ndarray:
        """
        Calculate confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalization mode ('true', 'pred', 'all', or None)

        Returns:
            Confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)

        logger.info(f"\nConfusion Matrix (normalize={normalize}):")
        logger.info(f"\n{cm}")

        return cm

    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        Get sklearn classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Classification report string
        """
        report = classification_report(y_true, y_pred, zero_division=0)

        logger.info("\nClassification Report:")
        logger.info(f"\n{report}")

        return report

    def calculate_roc_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        classes: Optional[list] = None
    ) -> Dict:
        """
        Calculate ROC curve and AUC for multi-class classification.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            classes: List of class names

        Returns:
            Dictionary with ROC curves and AUC scores
        """
        from sklearn.preprocessing import label_binarize

        if classes is None:
            classes = self.target_categories

        # Binarize labels
        y_true_bin = label_binarize(
            y_true,
            classes=classes
        )

        n_classes = len(classes)
        roc_metrics = {
            'fpr': {},
            'tpr': {},
            'auc': {},
            'classes': classes
        }

        # Calculate ROC curve and AUC for each class
        for i, class_name in enumerate(classes):
            if y_true_bin.shape[1] > 1:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
            else:
                # Binary case
                fpr, tpr, _ = roc_curve(y_true_bin, y_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)

            roc_metrics['fpr'][class_name] = fpr
            roc_metrics['tpr'][class_name] = tpr
            roc_metrics['auc'][class_name] = roc_auc

        # Calculate micro-average ROC curve and AUC
        if y_true_bin.shape[1] > 1:
            fpr_micro, tpr_micro, _ = roc_curve(
                y_true_bin.ravel(),
                y_pred_proba.ravel()
            )
            roc_auc_micro = auc(fpr_micro, tpr_micro)

            roc_metrics['fpr']['micro'] = fpr_micro
            roc_metrics['tpr']['micro'] = tpr_micro
            roc_metrics['auc']['micro'] = roc_auc_micro

        logger.info("\nAUC Scores:")
        for class_name, auc_score in roc_metrics['auc'].items():
            logger.info(f"  {class_name:15}: {auc_score:.4f}")

        return roc_metrics

    def calculate_pr_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        classes: Optional[list] = None
    ) -> Dict:
        """
        Calculate Precision-Recall curve for multi-class classification.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            classes: List of class names

        Returns:
            Dictionary with PR curves and average precision scores
        """
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import average_precision_score

        if classes is None:
            classes = self.target_categories

        # Binarize labels
        y_true_bin = label_binarize(
            y_true,
            classes=classes
        )

        pr_metrics = {
            'precision': {},
            'recall': {},
            'avg_precision': {},
            'classes': classes
        }

        # Calculate PR curve for each class
        for i, class_name in enumerate(classes):
            if y_true_bin.shape[1] > 1:
                precision, recall, _ = precision_recall_curve(
                    y_true_bin[:, i],
                    y_pred_proba[:, i]
                )
                avg_precision = average_precision_score(
                    y_true_bin[:, i],
                    y_pred_proba[:, i]
                )
            else:
                # Binary case
                precision, recall, _ = precision_recall_curve(
                    y_true_bin,
                    y_pred_proba[:, 1]
                )
                avg_precision = average_precision_score(
                    y_true_bin,
                    y_pred_proba[:, 1]
                )

            pr_metrics['precision'][class_name] = precision
            pr_metrics['recall'][class_name] = recall
            pr_metrics['avg_precision'][class_name] = avg_precision

        logger.info("\nAverage Precision Scores:")
        for class_name, ap_score in pr_metrics['avg_precision'].items():
            logger.info(f"  {class_name:15}: {ap_score:.4f}")

        return pr_metrics

    def generate_evaluation_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        model_name: str = "Model"
    ) -> Dict:
        """
        Generate comprehensive evaluation report.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            model_name: Name of the model

        Returns:
            Dictionary with all metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluation Report for {model_name}")
        logger.info(f"{'='*60}")

        report = {
            'model_name': model_name,
            'basic_metrics': {},
            'per_class_metrics': {},
            'confusion_matrix': None,
            'classification_report': None
        }

        # Basic metrics
        report['basic_metrics'] = self.calculate_basic_metrics(y_true, y_pred)

        # Per-class metrics
        report['per_class_metrics'] = self.calculate_per_class_metrics(y_true, y_pred)

        # Confusion matrix
        report['confusion_matrix'] = self.calculate_confusion_matrix(
            y_true, y_pred
        ).tolist()

        # Classification report
        report['classification_report'] = self.get_classification_report(
            y_true, y_pred
        )

        # ROC and PR metrics if probabilities are available
        if y_pred_proba is not None:
            try:
                report['roc_metrics'] = self.calculate_roc_metrics(
                    y_true, y_pred_proba
                )
                report['pr_metrics'] = self.calculate_pr_metrics(
                    y_true, y_pred_proba
                )
            except Exception as e:
                logger.warning(f"Could not calculate ROC/PR metrics: {e}")

        return report

    def save_report(
        self,
        report: Dict,
        output_path: str
    ):
        """
        Save evaluation report to JSON file.

        Args:
            report: Evaluation report dictionary
            output_path: Output file path
        """
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_report = convert_to_serializable(report)

        with open(output_path, 'w') as f:
            json.dump(serializable_report, f, indent=2)

        logger.info(f"\nReport saved to {output_path}")

    def compare_models(
        self,
        reports: Dict[str, Dict]
    ) -> Dict:
        """
        Compare multiple model reports.

        Args:
            reports: Dictionary of model reports

        Returns:
            Comparison summary
        """
        logger.info(f"\n{'='*60}")
        logger.info("Model Comparison")
        logger.info(f"{'='*60}\n")

        comparison = {}

        logger.info(f"{'Model':20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
        logger.info("-" * 65)

        for model_name, report in reports.items():
            metrics = report['basic_metrics']
            comparison[model_name] = metrics

            logger.info(
                f"{model_name:20} "
                f"{metrics['accuracy']:10.4f} "
                f"{metrics['precision']:10.4f} "
                f"{metrics['recall']:10.4f} "
                f"{metrics['f1_score']:10.4f}"
            )

        # Find best model for each metric
        logger.info(f"\n{'Best Models by Metric':20}")
        logger.info("-" * 40)

        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            best_model = max(comparison.items(), key=lambda x: x[1][metric])
            logger.info(f"  {metric:15}: {best_model[0]} ({best_model[1][metric]:.4f})")

        return comparison
