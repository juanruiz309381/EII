"""
Data augmentation module for audio data.

This module provides various audio augmentation techniques to prevent
overfitting and increase training data diversity.
"""

import numpy as np
import librosa
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioAugmenter:
    """
    Class for audio data augmentation.

    Implements various augmentation techniques including:
    - Time stretching
    - Pitch shifting
    - Adding noise
    - Time shifting
    """

    def __init__(self, config: dict):
        """
        Initialize AudioAugmenter.

        Args:
            config: Configuration dictionary with augmentation parameters
        """
        self.config = config
        self.augmentation_config = config['augmentation']
        self.enabled = self.augmentation_config['enabled']
        self.techniques = self.augmentation_config['techniques']
        self.sample_rate = config['audio']['sample_rate']

        logger.info(f"AudioAugmenter initialized (enabled: {self.enabled})")

    def time_stretch(
        self,
        audio: np.ndarray,
        rate: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply time stretching to audio.

        Args:
            audio: Audio signal
            rate: Stretch rate (1.0 = no change, <1 = slower, >1 = faster)
                 If None, randomly sampled from config range

        Returns:
            Time-stretched audio
        """
        if rate is None:
            rate_range = self.techniques['time_stretch']['rate_range']
            rate = np.random.uniform(rate_range[0], rate_range[1])

        augmented = librosa.effects.time_stretch(audio, rate=rate)

        # Ensure same length
        if len(augmented) < len(audio):
            augmented = np.pad(augmented, (0, len(audio) - len(augmented)))
        else:
            augmented = augmented[:len(audio)]

        return augmented

    def pitch_shift(
        self,
        audio: np.ndarray,
        n_steps: Optional[int] = None,
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply pitch shifting to audio.

        Args:
            audio: Audio signal
            n_steps: Number of semitones to shift
                    If None, randomly sampled from config range
            sr: Sample rate (uses config if None)

        Returns:
            Pitch-shifted audio
        """
        sr = sr or self.sample_rate

        if n_steps is None:
            n_steps_range = self.techniques['pitch_shift']['n_steps_range']
            n_steps = np.random.randint(n_steps_range[0], n_steps_range[1] + 1)

        augmented = librosa.effects.pitch_shift(
            audio,
            sr=sr,
            n_steps=n_steps
        )

        return augmented

    def add_noise(
        self,
        audio: np.ndarray,
        noise_factor: Optional[float] = None
    ) -> np.ndarray:
        """
        Add random noise to audio.

        Args:
            audio: Audio signal
            noise_factor: Noise intensity factor
                         If None, uses config value

        Returns:
            Audio with added noise
        """
        if noise_factor is None:
            noise_factor = self.techniques['add_noise']['noise_factor']

        noise = np.random.randn(len(audio))
        augmented = audio + noise_factor * noise

        # Normalize to prevent clipping
        augmented = augmented / np.max(np.abs(augmented))

        return augmented

    def time_shift(
        self,
        audio: np.ndarray,
        shift_max: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply time shifting to audio.

        Args:
            audio: Audio signal
            shift_max: Maximum shift as fraction of audio length
                      If None, uses config value

        Returns:
            Time-shifted audio
        """
        if shift_max is None:
            shift_max = self.techniques['time_shift']['shift_max']

        shift = np.random.randint(-int(len(audio) * shift_max),
                                   int(len(audio) * shift_max))

        augmented = np.roll(audio, shift)

        return augmented

    def apply_random_augmentation(
        self,
        audio: np.ndarray,
        techniques: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Apply random augmentation technique(s) to audio.

        Args:
            audio: Audio signal
            techniques: List of techniques to use
                       If None, randomly selects from enabled techniques

        Returns:
            Augmented audio
        """
        if not self.enabled:
            return audio

        # Get enabled techniques
        enabled_techniques = []

        if self.techniques['time_stretch']['enabled']:
            enabled_techniques.append('time_stretch')
        if self.techniques['pitch_shift']['enabled']:
            enabled_techniques.append('pitch_shift')
        if self.techniques['add_noise']['enabled']:
            enabled_techniques.append('add_noise')
        if self.techniques['time_shift']['enabled']:
            enabled_techniques.append('time_shift')

        if not enabled_techniques:
            return audio

        # Select techniques to apply
        if techniques is None:
            # Randomly select 1-2 techniques
            num_techniques = np.random.randint(1, 3)
            techniques = np.random.choice(
                enabled_techniques,
                size=min(num_techniques, len(enabled_techniques)),
                replace=False
            )

        # Apply selected techniques
        augmented = audio.copy()

        for technique in techniques:
            if technique == 'time_stretch':
                augmented = self.time_stretch(augmented)
            elif technique == 'pitch_shift':
                augmented = self.pitch_shift(augmented)
            elif technique == 'add_noise':
                augmented = self.add_noise(augmented)
            elif technique == 'time_shift':
                augmented = self.time_shift(augmented)

        return augmented

    def augment_batch(
        self,
        audio_list: List[np.ndarray],
        augmentation_factor: int = 1
    ) -> List[np.ndarray]:
        """
        Augment a batch of audio samples.

        Args:
            audio_list: List of audio signals
            augmentation_factor: Number of augmented versions per sample

        Returns:
            List of augmented audio samples (including originals)
        """
        if not self.enabled or augmentation_factor < 1:
            return audio_list

        augmented_list = []

        logger.info(f"Augmenting {len(audio_list)} samples with factor {augmentation_factor}...")

        for audio in audio_list:
            # Add original
            augmented_list.append(audio)

            # Add augmented versions
            for _ in range(augmentation_factor):
                augmented = self.apply_random_augmentation(audio)
                augmented_list.append(augmented)

        logger.info(f"Augmentation complete. Total samples: {len(augmented_list)}")

        return augmented_list

    def augment_dataset(
        self,
        audio_list: List[np.ndarray],
        labels: List[str],
        augmentation_factor: int = 1,
        balance_classes: bool = True
    ) -> tuple:
        """
        Augment entire dataset with optional class balancing.

        Args:
            audio_list: List of audio signals
            labels: List of labels
            augmentation_factor: Base augmentation factor
            balance_classes: Whether to balance classes through augmentation

        Returns:
            Tuple of (augmented_audio_list, augmented_labels)
        """
        if not self.enabled:
            return audio_list, labels

        from collections import Counter

        # Count samples per class
        class_counts = Counter(labels)
        max_count = max(class_counts.values())

        augmented_audio = []
        augmented_labels = []

        logger.info("Augmenting dataset...")
        logger.info(f"Original class distribution: {dict(class_counts)}")

        for audio, label in zip(audio_list, labels):
            # Add original
            augmented_audio.append(audio)
            augmented_labels.append(label)

            # Calculate augmentation factor for this sample
            if balance_classes:
                # Augment minority classes more
                current_count = class_counts[label]
                sample_aug_factor = int((max_count / current_count) * augmentation_factor)
            else:
                sample_aug_factor = augmentation_factor

            # Add augmented versions
            for _ in range(sample_aug_factor):
                aug_audio = self.apply_random_augmentation(audio)
                augmented_audio.append(aug_audio)
                augmented_labels.append(label)

        # Log new distribution
        new_class_counts = Counter(augmented_labels)
        logger.info(f"Augmented class distribution: {dict(new_class_counts)}")
        logger.info(f"Total samples: {len(augmented_audio)}")

        return augmented_audio, augmented_labels
