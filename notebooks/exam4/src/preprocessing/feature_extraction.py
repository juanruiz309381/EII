"""
Feature extraction module for audio data.

This module extracts various audio features including spectrograms,
MFCCs, and other acoustic features for classification.
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Class for extracting audio features.

    Supports multiple feature types including mel spectrograms,
    MFCCs, and conventional acoustic features.
    """

    def __init__(self, config: dict):
        """
        Initialize FeatureExtractor.

        Args:
            config: Configuration dictionary with feature parameters
        """
        self.config = config
        self.sample_rate = config['audio']['sample_rate']
        self.n_mels = config['audio']['n_mels']
        self.n_fft = config['audio']['n_fft']
        self.hop_length = config['audio']['hop_length']
        self.fmin = config['audio']['fmin']
        self.fmax = config['audio']['fmax']
        self.n_mfcc = config['features']['n_mfcc']
        self.image_size = tuple(config['features']['image_size'])

        logger.info("FeatureExtractor initialized")

    def extract_mel_spectrogram(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract mel spectrogram from audio.

        Args:
            audio: Audio signal
            sr: Sample rate (uses config if None)

        Returns:
            Mel spectrogram (in dB scale)
        """
        sr = sr or self.sample_rate

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax
        )

        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec_db

    def extract_mfcc(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract MFCC features from audio.

        Args:
            audio: Audio signal
            sr: Sample rate (uses config if None)

        Returns:
            MFCC features
        """
        sr = sr or self.sample_rate

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        return mfcc

    def extract_stft_spectrogram(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract STFT spectrogram from audio.

        Args:
            audio: Audio signal
            sr: Sample rate (uses config if None)

        Returns:
            STFT spectrogram (in dB scale)
        """
        sr = sr or self.sample_rate

        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Convert to dB scale
        stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

        return stft_db

    def extract_conventional_features(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Extract conventional acoustic features for classical ML models.

        Features include:
        - Zero Crossing Rate
        - Spectral Centroid
        - Spectral Rolloff
        - Spectral Bandwidth
        - RMS Energy
        - MFCC statistics

        Args:
            audio: Audio signal
            sr: Sample rate (uses config if None)

        Returns:
            Dictionary of features
        """
        sr = sr or self.sample_rate
        features = {}

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))

        # Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        features['spectral_centroid_std'] = float(np.std(spectral_centroids))

        # Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))

        # Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))

        # RMS Energy
        rms = librosa.feature.rms(y=audio)
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))

        # MFCC statistics
        mfcc = self.extract_mfcc(audio, sr)
        for i in range(self.n_mfcc):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfcc[i]))

        # Chroma features
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        features['chroma_mean'] = float(np.mean(chroma))
        features['chroma_std'] = float(np.std(chroma))

        return features

    def spectrogram_to_image(
        self,
        spectrogram: np.ndarray,
        resize: bool = True
    ) -> np.ndarray:
        """
        Convert spectrogram to image format suitable for deep learning.

        Args:
            spectrogram: Spectrogram data
            resize: Whether to resize to target image size

        Returns:
            Image array in RGB format
        """
        # Normalize to 0-255 range
        spec_normalized = ((spectrogram - spectrogram.min()) /
                          (spectrogram.max() - spectrogram.min()) * 255)
        spec_normalized = spec_normalized.astype(np.uint8)

        # Convert to RGB (3 channels)
        spec_rgb = cv2.applyColorMap(spec_normalized, cv2.COLORMAP_VIRIDIS)

        if resize:
            spec_rgb = cv2.resize(spec_rgb, self.image_size)

        # Normalize to 0-1 range for neural networks
        spec_rgb = spec_rgb.astype(np.float32) / 255.0

        return spec_rgb

    def save_spectrogram_image(
        self,
        spectrogram: np.ndarray,
        output_path: str,
        title: Optional[str] = None,
        sr: Optional[int] = None
    ):
        """
        Save spectrogram as image file.

        Args:
            spectrogram: Spectrogram data
            output_path: Output file path
            title: Plot title
            sr: Sample rate (uses config if None)
        """
        sr = sr or self.sample_rate

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            spectrogram,
            sr=sr,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis='mel',
            cmap='viridis'
        )
        plt.colorbar(format='%+2.0f dB')

        if title:
            plt.title(title)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Spectrogram saved to {output_path}")

    def extract_features_batch(
        self,
        audio_list: list,
        feature_type: str = 'mel'
    ) -> np.ndarray:
        """
        Extract features from a batch of audio files.

        Args:
            audio_list: List of audio signals
            feature_type: Type of features to extract
                         ('mel', 'mfcc', 'stft', 'conventional', 'image')

        Returns:
            Array of extracted features
        """
        features = []

        logger.info(f"Extracting {feature_type} features from {len(audio_list)} samples...")

        for audio in audio_list:
            if feature_type == 'mel':
                feat = self.extract_mel_spectrogram(audio)
            elif feature_type == 'mfcc':
                feat = self.extract_mfcc(audio)
            elif feature_type == 'stft':
                feat = self.extract_stft_spectrogram(audio)
            elif feature_type == 'conventional':
                feat_dict = self.extract_conventional_features(audio)
                feat = np.array(list(feat_dict.values()))
            elif feature_type == 'image':
                mel_spec = self.extract_mel_spectrogram(audio)
                feat = self.spectrogram_to_image(mel_spec)
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")

            features.append(feat)

        features = np.array(features)
        logger.info(f"Features extracted. Shape: {features.shape}")

        return features
