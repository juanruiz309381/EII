"""
Deep learning models for age classification with transfer learning.

This module implements CNN-based models using transfer learning from
pre-trained models (VGG16, ResNet50, MobileNetV2, EfficientNet).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    VGG16, ResNet50, MobileNetV2, EfficientNetB0
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from sklearn.preprocessing import LabelEncoder
import logging
from typing import Tuple, Optional, Dict
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepLearningModel:
    """
    Class for deep learning models with transfer learning.

    Implements CNN-based classification with pre-trained models
    and anti-overfitting techniques.
    """

    def __init__(self, config: dict):
        """
        Initialize DeepLearningModel.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.dl_config = config['deep_learning']
        self.image_size = tuple(config['features']['image_size'])
        self.num_classes = len(config['data']['target_categories'])
        self.target_categories = config['data']['target_categories']

        self.model = None
        self.history = None
        self.label_encoder = LabelEncoder()

        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

        logger.info("DeepLearningModel initialized")
        logger.info(f"Image size: {self.image_size}")
        logger.info(f"Number of classes: {self.num_classes}")

    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_encoder: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.

        Args:
            X: Image data
            y: Labels
            fit_encoder: Whether to fit the label encoder

        Returns:
            Tuple of (images, encoded_labels)
        """
        # Ensure proper shape (N, H, W, C)
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=-1)

        # Ensure images are in [0, 1] range
        if X.max() > 1.0:
            X = X / 255.0

        # Encode labels
        if fit_encoder:
            self.label_encoder.fit(y)

        y_encoded = self.label_encoder.transform(y)

        # Convert to one-hot encoding
        y_onehot = keras.utils.to_categorical(y_encoded, num_classes=self.num_classes)

        logger.info(f"Data prepared: X shape {X.shape}, y shape {y_onehot.shape}")

        return X, y_onehot

    def create_transfer_learning_model(
        self,
        base_model_name: str = 'VGG16',
        trainable_layers: int = 4
    ) -> keras.Model:
        """
        Create model using transfer learning.

        Args:
            base_model_name: Name of pre-trained model to use
            trainable_layers: Number of layers to unfreeze for fine-tuning

        Returns:
            Keras model
        """
        logger.info(f"\nCreating model with {base_model_name} as base")

        # Get base model
        if base_model_name == 'VGG16':
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3)
            )
        elif base_model_name == 'ResNet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3)
            )
        elif base_model_name == 'MobileNetV2':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3)
            )
        elif base_model_name == 'EfficientNetB0':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.image_size, 3)
            )
        else:
            raise ValueError(f"Unknown base model: {base_model_name}")

        # Freeze base model layers
        base_model.trainable = False

        # Create new model
        inputs = keras.Input(shape=(*self.image_size, 3))

        # Preprocessing for pre-trained models
        x = inputs

        # Base model
        x = base_model(x, training=False)

        # Add custom layers
        x = layers.GlobalAveragePooling2D()(x)

        # Add regularization
        if self.dl_config['regularization']['batch_normalization']:
            x = layers.BatchNormalization()(x)

        # Dense layers with dropout
        dropout_rate = self.dl_config['regularization']['dropout_rate']
        l2_reg = self.dl_config['regularization']['l2_regularization']

        x = layers.Dense(
            512,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        )(x)

        if self.dl_config['regularization']['batch_normalization']:
            x = layers.BatchNormalization()(x)

        x = layers.Dropout(dropout_rate)(x)

        x = layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        )(x)

        if self.dl_config['regularization']['batch_normalization']:
            x = layers.BatchNormalization()(x)

        x = layers.Dropout(dropout_rate)(x)

        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='predictions'
        )(x)

        # Create model
        model = keras.Model(inputs, outputs, name=f'{base_model_name}_transfer_learning')

        # Optionally unfreeze last few layers for fine-tuning
        if trainable_layers > 0:
            logger.info(f"Unfreezing last {trainable_layers} layers for fine-tuning")
            for layer in base_model.layers[-trainable_layers:]:
                layer.trainable = True

        logger.info(f"Model created successfully")
        logger.info(f"Total parameters: {model.count_params():,}")
        logger.info(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

        return model

    def create_custom_cnn(self) -> keras.Model:
        """
        Create custom CNN model without transfer learning.

        Returns:
            Keras model
        """
        logger.info("\nCreating custom CNN model")

        cnn_config = self.dl_config['custom_cnn']
        dropout_rate = self.dl_config['regularization']['dropout_rate']
        l2_reg = self.dl_config['regularization']['l2_regularization']
        use_bn = self.dl_config['regularization']['batch_normalization']

        inputs = keras.Input(shape=(*self.image_size, 3))
        x = inputs

        # Convolutional blocks
        for i, filters in enumerate(cnn_config['filters']):
            x = layers.Conv2D(
                filters,
                cnn_config['kernel_size'],
                padding='same',
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(l2_reg),
                name=f'conv_{i}'
            )(x)

            if use_bn:
                x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(cnn_config['pool_size'])(x)
            x = layers.Dropout(dropout_rate / 2)(x)

        # Flatten and dense layers
        x = layers.Flatten()(x)

        for i, units in enumerate(cnn_config['dense_units']):
            x = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(l2_reg),
                name=f'dense_{i}'
            )(x)

            if use_bn:
                x = layers.BatchNormalization()(x)

            x = layers.Dropout(dropout_rate)(x)

        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='predictions'
        )(x)

        model = keras.Model(inputs, outputs, name='custom_cnn')

        logger.info(f"Custom CNN created successfully")
        logger.info(f"Total parameters: {model.count_params():,}")

        return model

    def compile_model(
        self,
        model: keras.Model,
        learning_rate: Optional[float] = None
    ):
        """
        Compile model with optimizer and loss.

        Args:
            model: Keras model
            learning_rate: Learning rate (uses config if None)
        """
        lr = learning_rate or self.dl_config['training']['learning_rate']
        optimizer_name = self.dl_config['training']['optimizer']

        if optimizer_name == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=lr)
        elif optimizer_name == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )

        self.model = model
        logger.info(f"Model compiled with {optimizer_name} optimizer (lr={lr})")

    def get_callbacks(self, output_dir: str) -> list:
        """
        Get training callbacks.

        Args:
            output_dir: Directory for saving checkpoints and logs

        Returns:
            List of callbacks
        """
        callbacks = []

        # Early stopping
        if self.dl_config['training']['early_stopping']['enabled']:
            es_config = self.dl_config['training']['early_stopping']
            callbacks.append(
                EarlyStopping(
                    monitor='val_loss',
                    patience=es_config['patience'],
                    min_delta=es_config['min_delta'],
                    restore_best_weights=es_config['restore_best_weights'],
                    verbose=1
                )
            )
            logger.info("Early stopping enabled")

        # Learning rate schedule
        if self.dl_config['training']['lr_schedule']['enabled']:
            lr_config = self.dl_config['training']['lr_schedule']
            callbacks.append(
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=lr_config['factor'],
                    patience=lr_config['patience'],
                    min_lr=lr_config['min_lr'],
                    verbose=1
                )
            )
            logger.info("Learning rate scheduling enabled")

        # Model checkpoint
        if self.dl_config['training']['checkpoint']['enabled']:
            os.makedirs(output_dir, exist_ok=True)
            checkpoint_path = os.path.join(output_dir, 'best_model.h5')
            callbacks.append(
                ModelCheckpoint(
                    checkpoint_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            )
            logger.info(f"Model checkpointing enabled: {checkpoint_path}")

        # TensorBoard
        tensorboard_dir = os.path.join(output_dir, 'tensorboard_logs')
        os.makedirs(tensorboard_dir, exist_ok=True)
        callbacks.append(
            TensorBoard(
                log_dir=tensorboard_dir,
                histogram_freq=1
            )
        )
        logger.info(f"TensorBoard logging enabled: {tensorboard_dir}")

        return callbacks

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        output_dir: str = 'results/models'
    ) -> keras.callbacks.History:
        """
        Train the model.

        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            output_dir: Directory for saving checkpoints

        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_*_model() first.")

        logger.info("\n" + "="*60)
        logger.info("Starting training")
        logger.info("="*60)

        # Prepare data
        X_train_prep, y_train_prep = self.prepare_data(X_train, y_train, fit_encoder=True)
        X_val_prep, y_val_prep = self.prepare_data(X_val, y_val, fit_encoder=False)

        # Get callbacks
        callbacks = self.get_callbacks(output_dir)

        # Train
        batch_size = self.dl_config['training']['batch_size']
        epochs = self.dl_config['training']['epochs']

        self.history = self.model.fit(
            X_train_prep,
            y_train_prep,
            validation_data=(X_val_prep, y_val_prep),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        logger.info("\nTraining completed!")

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Images

        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained")

        # Prepare data
        X_prep, _ = self.prepare_data(X, np.zeros(len(X)), fit_encoder=False)

        # Predict
        y_pred_proba = self.model.predict(X_prep)
        y_pred_encoded = np.argmax(y_pred_proba, axis=1)

        # Decode labels
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Images

        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained")

        # Prepare data
        X_prep, _ = self.prepare_data(X, np.zeros(len(X)), fit_encoder=False)

        # Predict probabilities
        y_proba = self.model.predict(X_prep)

        return y_proba

    def save_model(self, output_path: str):
        """
        Save trained model.

        Args:
            output_path: Output file path
        """
        if self.model is None:
            raise ValueError("Model not trained")

        self.model.save(output_path)
        logger.info(f"Model saved to {output_path}")

    def load_model(self, model_path: str):
        """
        Load trained model.

        Args:
            model_path: Path to model file
        """
        self.model = keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
