#!/usr/bin/env python3
"""
Main training script for Age Recognition from Voice.

This script orchestrates the entire training pipeline including:
- Data loading and preprocessing
- Feature extraction
- Data augmentation
- Model training (conventional and deep learning)
- Evaluation and visualization
"""

import os
import sys
import yaml
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Configure TensorFlow GPU memory growth to avoid OOM
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
try:
    import tensorflow as tf
    # Allow memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        logging.info(f"✅ GPU memory growth enabled for {len(physical_devices)} device(s)")
except Exception as e:
    logging.warning(f"Could not configure GPU memory growth: {e}")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import AudioLoader, FeatureExtractor, AudioAugmenter
from models import ConventionalModels, DeepLearningModel
from utils import MetricsCalculator, Visualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_and_prepare_data(config: dict, max_samples: int = None):
    """
    Load and prepare dataset.

    Args:
        config: Configuration dictionary
        max_samples: Maximum samples per category (None for all)

    Returns:
        Tuple of data splits
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Loading and Preparing Data")
    logger.info("="*60)

    # Initialize loader
    audio_loader = AudioLoader(config)

    # Get dataset statistics
    stats = audio_loader.get_dataset_statistics()
    logger.info("\nDataset Statistics:")
    logger.info(f"Total files: {stats['total_files']}")
    logger.info(f"Target category distribution: {stats['target_category_counts']}")

    # Load dataset
    audio_data, labels, file_paths = audio_loader.load_dataset(
        max_samples_per_category=max_samples,
        shuffle=True
    )

    # Split data
    split_config = config['split']

    # First split: train and temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        audio_data,
        labels,
        test_size=(split_config['validation'] + split_config['test']),
        random_state=split_config['random_state'],
        stratify=labels if split_config['stratify'] else None
    )

    # Second split: temp into val and test
    val_ratio = split_config['validation'] / (split_config['validation'] + split_config['test'])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1 - val_ratio),
        random_state=split_config['random_state'],
        stratify=y_temp if split_config['stratify'] else None
    )

    logger.info("\nData Split:")
    logger.info(f"  Training:   {len(X_train)} samples")
    logger.info(f"  Validation: {len(X_val)} samples")
    logger.info(f"  Test:       {len(X_test)} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_augmentation(config: dict, X_train, y_train):
    """Apply data augmentation to training data."""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Applying Data Augmentation")
    logger.info("="*60)

    augmenter = AudioAugmenter(config)

    if augmenter.enabled:
        X_train_aug, y_train_aug = augmenter.augment_dataset(
            X_train,
            y_train,
            augmentation_factor=1,
            balance_classes=True
        )
        return X_train_aug, y_train_aug
    else:
        logger.info("Data augmentation disabled")
        return X_train, y_train


def extract_features(config: dict, X_train, X_val, X_test, feature_type: str):
    """Extract features from audio data."""
    logger.info("\n" + "="*60)
    logger.info(f"STEP 3: Extracting Features ({feature_type})")
    logger.info("="*60)

    extractor = FeatureExtractor(config)

    X_train_features = extractor.extract_features_batch(X_train, feature_type=feature_type)
    X_val_features = extractor.extract_features_batch(X_val, feature_type=feature_type)
    X_test_features = extractor.extract_features_batch(X_test, feature_type=feature_type)

    logger.info("\nFeature extraction completed:")
    logger.info(f"  Train features: {X_train_features.shape}")
    logger.info(f"  Val features:   {X_val_features.shape}")
    logger.info(f"  Test features:  {X_test_features.shape}")

    return X_train_features, X_val_features, X_test_features


def train_conventional_models(
    config: dict,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
):
    """Train conventional ML models."""
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Training Conventional Models")
    logger.info("="*60)

    # Extract conventional features
    X_train_conv, X_val_conv, X_test_conv = extract_features(
        config, X_train, X_val, X_test, feature_type='conventional'
    )

    # Initialize models
    conv_models = ConventionalModels(config)

    # Train all models
    trained_models = conv_models.train_all_models(
        X_train_conv,
        y_train,
        X_val_conv,
        y_val
    )

    # Evaluate and save results
    results = {}
    metrics_calc = MetricsCalculator(config)
    visualizer = Visualizer(config)

    for model_name in trained_models.keys():
        logger.info(f"\nEvaluating {model_name}...")

        # Predictions
        y_pred = conv_models.predict(model_name, X_test_conv)
        y_pred_proba = conv_models.predict_proba(model_name, X_test_conv)

        # Generate report
        report = metrics_calc.generate_evaluation_report(
            y_test,
            y_pred,
            y_pred_proba,
            model_name=model_name
        )

        results[model_name] = report

        # Save model
        model_path = os.path.join(config['data']['output_path'], 'models', f'{model_name}.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        conv_models.save_model(model_name, model_path)

        # Save report
        report_path = os.path.join(config['data']['output_path'], 'metrics', f'{model_name}_report.json')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        metrics_calc.save_report(report, report_path)

        # Generate plots
        feature_importance = conv_models.get_feature_importance(model_name)
        visualizer.generate_all_plots(
            y_test,
            y_pred,
            y_pred_proba,
            feature_importance=feature_importance,
            model_name=model_name,
            output_dir=os.path.join(config['data']['output_path'], 'plots', model_name)
        )

    return results


def train_deep_learning_model(
    config: dict,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
):
    """Train deep learning model with transfer learning."""
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Training Deep Learning Model")
    logger.info("="*60)

    # Extract image features (spectrograms)
    X_train_img, X_val_img, X_test_img = extract_features(
        config, X_train, X_val, X_test, feature_type='image'
    )

    # Initialize model
    dl_model = DeepLearningModel(config)

    # Create model with transfer learning
    if config['deep_learning']['transfer_learning']['enabled']:
        base_model = config['deep_learning']['transfer_learning']['base_model']
        trainable_layers = config['deep_learning']['transfer_learning']['trainable_layers']

        model = dl_model.create_transfer_learning_model(
            base_model_name=base_model,
            trainable_layers=trainable_layers
        )
    else:
        model = dl_model.create_custom_cnn()

    # Compile model
    dl_model.compile_model(model)

    # Print model summary
    logger.info("\nModel Summary:")
    model.summary(print_fn=logger.info)

    # Train model
    output_dir = os.path.join(config['data']['output_path'], 'models', 'deep_learning')
    history = dl_model.train(
        X_train_img,
        y_train,
        X_val_img,
        y_val,
        output_dir=output_dir
    )

    # Evaluate model
    logger.info("\nEvaluating deep learning model...")

    y_pred = dl_model.predict(X_test_img)
    y_pred_proba = dl_model.predict_proba(X_test_img)

    # Generate report
    metrics_calc = MetricsCalculator(config)
    report = metrics_calc.generate_evaluation_report(
        y_test,
        y_pred,
        y_pred_proba,
        model_name='deep_learning'
    )

    # Save model
    model_path = os.path.join(output_dir, 'final_model.h5')
    dl_model.save_model(model_path)

    # Save report
    report_path = os.path.join(config['data']['output_path'], 'metrics', 'deep_learning_report.json')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    metrics_calc.save_report(report, report_path)

    # Generate plots
    visualizer = Visualizer(config)
    visualizer.generate_all_plots(
        y_test,
        y_pred,
        y_pred_proba,
        history=history.history,
        model_name='deep_learning',
        output_dir=os.path.join(config['data']['output_path'], 'plots', 'deep_learning')
    )

    # Plot sample spectrograms
    visualizer.plot_spectrogram_samples(
        X_test_img[:8],
        y_test[:8],
        title='Sample Spectrograms',
        output_path=os.path.join(config['data']['output_path'], 'plots', 'sample_spectrograms.png')
    )

    return report


def compare_all_models(config: dict, results: dict):
    """Compare all trained models."""
    logger.info("\n" + "="*60)
    logger.info("STEP 6: Model Comparison")
    logger.info("="*60)

    metrics_calc = MetricsCalculator(config)
    visualizer = Visualizer(config)

    # Compare models
    comparison = metrics_calc.compare_models(results)

    # Plot comparison
    visualizer.plot_model_comparison(
        comparison,
        title='Model Performance Comparison',
        output_path=os.path.join(config['data']['output_path'], 'plots', 'model_comparison.png')
    )

    return comparison


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train age recognition models')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples per category (for quick testing)'
    )
    parser.add_argument(
        '--skip-conventional',
        action='store_true',
        help='Skip training conventional models'
    )
    parser.add_argument(
        '--skip-deep-learning',
        action='store_true',
        help='Skip training deep learning model'
    )

    args = parser.parse_args()

    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(args.config)

    # Create output directories
    os.makedirs(config['data']['output_path'], exist_ok=True)

    try:
        # Load and prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data(
            config,
            max_samples=args.max_samples
        )

        # Apply augmentation
        X_train, y_train = apply_augmentation(config, X_train, y_train)

        # Initialize visualizer
        visualizer = Visualizer(config)

        # Plot class distribution
        visualizer.plot_class_distribution(
            y_train,
            title='Training Set Class Distribution',
            output_path=os.path.join(config['data']['output_path'], 'plots', 'class_distribution.png')
        )

        # Store results
        all_results = {}

        # Train conventional models
        if not args.skip_conventional:
            conv_results = train_conventional_models(
                config,
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test
            )
            all_results.update(conv_results)

        # Train deep learning model
        if not args.skip_deep_learning:
            dl_result = train_deep_learning_model(
                config,
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test
            )
            all_results['deep_learning'] = dl_result

        # Compare all models
        if len(all_results) > 1:
            comparison = compare_all_models(config, all_results)

        logger.info("\n" + "="*60)
        logger.info("Training pipeline completed successfully!")
        logger.info("="*60)
        logger.info(f"\nResults saved to: {config['data']['output_path']}")

    except Exception as e:
        logger.error(f"\nError during training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
