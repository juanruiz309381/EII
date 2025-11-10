#!/usr/bin/env python3
"""
Script de predicción para reconocimiento de edad por voz.

Uso:
    python predict.py --audio mi_audio.mp3 --model random_forest
    python predict.py --audio mi_audio.mp3 --model deep_learning
"""

import os
import sys
import yaml
import argparse
import numpy as np
import joblib
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import AudioLoader, FeatureExtractor
from models import DeepLearningModel


def load_config(config_path='config/config.yaml'):
    """Cargar configuración."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_conventional_model(model_name, model_path):
    """
    Cargar modelo convencional.

    Args:
        model_name: Nombre del modelo
        model_path: Ruta al archivo del modelo

    Returns:
        Dictionary con modelo, scaler y label_encoder
    """
    print(f"📦 Cargando modelo {model_name}...")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    model_data = joblib.load(model_path)
    print(f"✅ Modelo cargado exitosamente")

    return model_data


def load_deep_learning_model(config, model_path):
    """
    Cargar modelo de Deep Learning.

    Args:
        config: Configuración
        model_path: Ruta al modelo

    Returns:
        Modelo de Deep Learning
    """
    print(f"📦 Cargando modelo Deep Learning...")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    from tensorflow import keras

    # Cargar el modelo
    model_keras = keras.models.load_model(model_path)

    # Crear wrapper
    dl_model = DeepLearningModel(config)
    dl_model.model = model_keras

    # Cargar label encoder (asumimos que está guardado)
    encoder_path = os.path.join(os.path.dirname(model_path), 'label_encoder.pkl')
    if os.path.exists(encoder_path):
        dl_model.label_encoder = joblib.load(encoder_path)
    else:
        # Crear encoder con categorías por defecto
        from sklearn.preprocessing import LabelEncoder
        dl_model.label_encoder = LabelEncoder()
        dl_model.label_encoder.fit(config['data']['target_categories'])

    print(f"✅ Modelo cargado exitosamente")

    return dl_model


def process_audio(audio_path, config, feature_type='conventional'):
    """
    Procesar archivo de audio.

    Args:
        audio_path: Ruta al archivo de audio
        config: Configuración
        feature_type: Tipo de features ('conventional' o 'image')

    Returns:
        Features extraídas
    """
    print(f"\n🎵 Procesando audio: {audio_path}")

    # Verificar que el archivo existe
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio no encontrado: {audio_path}")

    # Cargar audio
    audio_loader = AudioLoader(config)
    audio, sr = audio_loader.load_audio_file(audio_path)

    if audio is None:
        raise ValueError(f"Error al cargar el audio: {audio_path}")

    print(f"✅ Audio cargado ({len(audio)} samples, {sr} Hz)")

    # Extraer características
    extractor = FeatureExtractor(config)

    if feature_type == 'conventional':
        print("🔍 Extrayendo características acústicas...")
        features_dict = extractor.extract_conventional_features(audio, sr)
        features = np.array(list(features_dict.values()))
        features = features.reshape(1, -1)  # Reshape para predicción

    elif feature_type == 'image':
        print("🔍 Generando espectrograma...")
        mel_spec = extractor.extract_mel_spectrogram(audio, sr)
        features = extractor.spectrogram_to_image(mel_spec, resize=True)
        features = np.expand_dims(features, axis=0)  # Add batch dimension

    else:
        raise ValueError(f"Tipo de features no válido: {feature_type}")

    print(f"✅ Features extraídas - Shape: {features.shape}")

    return features


def predict_conventional(model_data, features):
    """
    Hacer predicción con modelo convencional.

    Args:
        model_data: Datos del modelo (model, scaler, label_encoder)
        features: Features del audio

    Returns:
        Tuple (predicción, probabilidades)
    """
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoder = model_data['label_encoder']

    # Escalar features
    features_scaled = scaler.transform(features)

    # Predecir
    y_pred_encoded = model.predict(features_scaled)
    y_pred_proba = model.predict_proba(features_scaled)

    # Decodificar
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    return y_pred[0], y_pred_proba[0]


def predict_deep_learning(dl_model, features):
    """
    Hacer predicción con modelo Deep Learning.

    Args:
        dl_model: Modelo de Deep Learning
        features: Features del audio (imagen)

    Returns:
        Tuple (predicción, probabilidades)
    """
    y_pred = dl_model.predict(features)
    y_pred_proba = dl_model.predict_proba(features)

    return y_pred[0], y_pred_proba[0]


def print_prediction(prediction, probabilities, config):
    """
    Imprimir resultado de predicción.

    Args:
        prediction: Predicción (categoría)
        probabilities: Probabilidades por clase
        config: Configuración
    """
    categories = config['data']['target_categories']

    print("\n" + "="*60)
    print("📊 RESULTADO DE LA PREDICCIÓN")
    print("="*60)

    print(f"\n🎯 Predicción: {prediction.upper()}")
    print(f"   Confianza: {max(probabilities)*100:.2f}%\n")

    print("📈 Probabilidades por categoría:")
    print("-" * 60)

    # Ordenar por probabilidad
    probs_with_cats = list(zip(categories, probabilities))
    probs_with_cats.sort(key=lambda x: x[1], reverse=True)

    for category, prob in probs_with_cats:
        bar_length = int(prob * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        marker = "👉" if category == prediction else "  "
        print(f"{marker} {category:15} [{bar}] {prob*100:6.2f}%")

    print("="*60 + "\n")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Predicción de edad por voz',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Usar Random Forest
  python predict.py --audio mi_audio.mp3 --model random_forest

  # Usar SVM
  python predict.py --audio voz.wav --model svm

  # Usar Deep Learning
  python predict.py --audio audio.mp3 --model deep_learning

  # Especificar ruta de modelo personalizada
  python predict.py --audio audio.mp3 --model random_forest --model-path mi_modelo.pkl

Modelos disponibles:
  - random_forest
  - svm
  - xgboost
  - deep_learning
        """
    )

    parser.add_argument(
        '--audio',
        type=str,
        required=True,
        help='Ruta al archivo de audio (MP3, WAV, etc.)'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['random_forest', 'svm', 'xgboost', 'deep_learning'],
        help='Modelo a utilizar'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Ruta personalizada al modelo (opcional)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Ruta al archivo de configuración'
    )

    args = parser.parse_args()

    try:
        # Cargar configuración
        print("⚙️  Cargando configuración...")
        config = load_config(args.config)
        print("✅ Configuración cargada\n")

        # Determinar ruta del modelo
        if args.model_path:
            model_path = args.model_path
        else:
            if args.model == 'deep_learning':
                model_path = 'results/models/deep_learning/best_model.h5'
                if not os.path.exists(model_path):
                    model_path = 'results/models/deep_learning/final_model.h5'
            else:
                model_path = f'results/models/{args.model}.pkl'

        # Cargar modelo
        if args.model == 'deep_learning':
            model = load_deep_learning_model(config, model_path)
            feature_type = 'image'
        else:
            model = load_conventional_model(args.model, model_path)
            feature_type = 'conventional'

        # Procesar audio
        features = process_audio(args.audio, config, feature_type=feature_type)

        # Hacer predicción
        print("\n🤖 Haciendo predicción...")

        if args.model == 'deep_learning':
            prediction, probabilities = predict_deep_learning(model, features)
        else:
            prediction, probabilities = predict_conventional(model, features)

        # Mostrar resultado
        print_prediction(prediction, probabilities, config)

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
