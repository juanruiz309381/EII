"""
Conventional machine learning models for age classification.

This module implements classical ML algorithms including:
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import logging
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConventionalModels:
    """
    Class for conventional machine learning models.

    Implements training and evaluation of classical ML algorithms
    on acoustic features.
    """

    def __init__(self, config: dict):
        """
        Initialize ConventionalModels.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models_config = config['conventional_models']
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = None

        logger.info("ConventionalModels initialized")

    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.

        Args:
            X: Feature matrix
            y: Labels
            fit_scaler: Whether to fit the scaler (True for training)

        Returns:
            Tuple of (scaled_features, encoded_labels)
        """
        # Reshape if needed (for spectrograms)
        if len(X.shape) > 2:
            n_samples = X.shape[0]
            X = X.reshape(n_samples, -1)

        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
            self.label_encoder.fit(y)
        else:
            X_scaled = self.scaler.transform(X)

        # Encode labels
        y_encoded = self.label_encoder.transform(y)

        logger.info(f"Data prepared: X shape {X_scaled.shape}, y shape {y_encoded.shape}")

        return X_scaled, y_encoded

    def create_random_forest(self, params: Dict) -> RandomForestClassifier:
        """
        Create Random Forest classifier.

        Args:
            params: Model parameters

        Returns:
            RandomForestClassifier instance
        """
        model = RandomForestClassifier(**params)
        logger.info(f"Random Forest created with params: {params}")
        return model

    def create_svm(self, params: Dict) -> SVC:
        """
        Create SVM classifier.

        Args:
            params: Model parameters

        Returns:
            SVC instance
        """
        # Add probability=True for probability estimates
        params['probability'] = True
        model = SVC(**params)
        logger.info(f"SVM created with params: {params}")
        return model

    def create_xgboost(self, params: Dict) -> xgb.XGBClassifier:
        """
        Create XGBoost classifier.

        Args:
            params: Model parameters

        Returns:
            XGBClassifier instance
        """
        model = xgb.XGBClassifier(**params)
        logger.info(f"XGBoost created with params: {params}")
        return model

    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> object:
        """
        Train a specific model.

        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Trained model
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name} model")
        logger.info(f"{'='*60}")

        # Get model configuration
        model_config = next(
            (m for m in self.models_config if m['name'] == model_name),
            None
        )

        if model_config is None:
            raise ValueError(f"Model {model_name} not found in configuration")

        params = model_config['params']

        # Create model
        if model_name == 'random_forest':
            model = self.create_random_forest(params)
        elif model_name == 'svm':
            model = self.create_svm(params)
        elif model_name == 'xgboost':
            model = self.create_xgboost(params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Prepare training data
        X_train_scaled, y_train_encoded = self.prepare_data(
            X_train, y_train, fit_scaler=True
        )

        # Train model
        logger.info("Training started...")

        if model_name == 'xgboost' and X_val is not None and y_val is not None:
            # Use early stopping for XGBoost
            X_val_scaled, y_val_encoded = self.prepare_data(
                X_val, y_val, fit_scaler=False
            )

            model.fit(
                X_train_scaled,
                y_train_encoded,
                eval_set=[(X_val_scaled, y_val_encoded)],
                verbose=False
            )
        else:
            model.fit(X_train_scaled, y_train_encoded)

        logger.info("Training completed!")

        # Cross-validation score
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train_encoded, cv=5, scoring='accuracy'
        )
        logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Store model
        self.models[model_name] = model

        return model

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, object]:
        """
        Train all configured models.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Dictionary of trained models
        """
        logger.info("\n" + "="*60)
        logger.info("Training all conventional models")
        logger.info("="*60 + "\n")

        for model_config in self.models_config:
            model_name = model_config['name']
            try:
                self.train_model(model_name, X_train, y_train, X_val, y_val)
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")

        return self.models

    def predict(
        self,
        model_name: str,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Make predictions with a trained model.

        Args:
            model_name: Name of the model to use
            X: Features

        Returns:
            Predicted labels
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")

        model = self.models[model_name]

        # Prepare data
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1))

        # Predict
        y_pred_encoded = model.predict(X_scaled)

        # Decode labels
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

        return y_pred

    def predict_proba(
        self,
        model_name: str,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            model_name: Name of the model to use
            X: Features

        Returns:
            Prediction probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")

        model = self.models[model_name]

        # Prepare data
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1))

        # Predict probabilities
        y_proba = model.predict_proba(X_scaled)

        return y_proba

    def get_feature_importance(
        self,
        model_name: str
    ) -> Optional[np.ndarray]:
        """
        Get feature importance for tree-based models.

        Args:
            model_name: Name of the model

        Returns:
            Feature importance array or None
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")

        model = self.models[model_name]

        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        else:
            logger.warning(f"Model {model_name} does not have feature importance")
            return None

    def save_model(
        self,
        model_name: str,
        output_path: str
    ):
        """
        Save trained model to file.

        Args:
            model_name: Name of the model to save
            output_path: Output file path
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")

        # Save model, scaler, and label encoder
        model_data = {
            'model': self.models[model_name],
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }

        joblib.dump(model_data, output_path)
        logger.info(f"Model saved to {output_path}")

    def load_model(
        self,
        model_name: str,
        model_path: str
    ):
        """
        Load trained model from file.

        Args:
            model_name: Name to assign to the loaded model
            model_path: Path to model file
        """
        model_data = joblib.load(model_path)

        self.models[model_name] = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']

        logger.info(f"Model loaded from {model_path}")
