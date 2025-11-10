"""
Audio loading and organization module.

This module handles loading audio files, organizing them by age categories,
and preparing them for feature extraction.
"""

import os
import glob
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioLoader:
    """
    Class for loading and organizing audio files.

    This class maps Common Voice age categories to the project's
    target categories and loads audio files accordingly.
    """

    def __init__(self, config: dict):
        """
        Initialize AudioLoader.

        Args:
            config: Configuration dictionary with data paths and category mapping
        """
        self.config = config
        self.base_path = config['data']['base_path']
        self.category_mapping = config['data']['category_mapping']
        self.target_categories = config['data']['target_categories']
        self.sample_rate = config['audio']['sample_rate']
        self.duration = config['audio']['duration']

        logger.info("AudioLoader initialized")
        logger.info(f"Base path: {self.base_path}")
        logger.info(f"Target categories: {self.target_categories}")

    def load_audio_file(
        self,
        file_path: str,
        sr: Optional[int] = None,
        duration: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Load a single audio file.

        Args:
            file_path: Path to audio file
            sr: Target sample rate (uses config if None)
            duration: Target duration in seconds (uses config if None)

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        sr = sr or self.sample_rate
        duration = duration or self.duration

        try:
            # Load audio
            audio, _ = librosa.load(file_path, sr=sr, duration=duration)

            # Ensure fixed length
            target_length = int(sr * duration)

            if len(audio) < target_length:
                # Pad if too short
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            else:
                # Trim if too long
                audio = audio[:target_length]

            return audio, sr

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None, None

    def get_category_mapping(self) -> Dict[str, str]:
        """
        Get the mapping from original to target categories.

        Returns:
            Dictionary mapping original to target categories
        """
        return {k: v for k, v in self.category_mapping.items() if v is not None}

    def load_dataset(
        self,
        max_samples_per_category: Optional[int] = None,
        shuffle: bool = True
    ) -> Tuple[List[np.ndarray], List[str], List[str]]:
        """
        Load all audio files organized by category.

        Args:
            max_samples_per_category: Maximum samples per category (None for all)
            shuffle: Whether to shuffle the files within each category

        Returns:
            Tuple of (audio_list, label_list, file_paths)
        """
        audio_data = []
        labels = []
        file_paths = []

        category_mapping = self.get_category_mapping()

        # Count samples per target category
        category_counts = Counter()

        logger.info("Loading audio files...")

        for original_category, target_category in category_mapping.items():
            if target_category is None:
                continue

            category_path = os.path.join(self.base_path, original_category)

            if not os.path.exists(category_path):
                logger.warning(f"Category path not found: {category_path}")
                continue

            # Get all audio files
            audio_files = glob.glob(os.path.join(category_path, "*.mp3"))

            if shuffle:
                np.random.shuffle(audio_files)

            # Limit samples if specified
            if max_samples_per_category:
                audio_files = audio_files[:max_samples_per_category]

            logger.info(f"Loading {len(audio_files)} files from {original_category} -> {target_category}")

            for file_path in audio_files:
                audio, sr = self.load_audio_file(file_path)

                if audio is not None:
                    audio_data.append(audio)
                    labels.append(target_category)
                    file_paths.append(file_path)
                    category_counts[target_category] += 1

        logger.info("\nDataset loaded successfully!")
        logger.info(f"Total samples: {len(audio_data)}")
        logger.info("\nSamples per category:")
        for category, count in sorted(category_counts.items()):
            logger.info(f"  {category:15}: {count:5} samples")

        return audio_data, labels, file_paths

    def get_dataset_statistics(self) -> Dict:
        """
        Get statistics about the dataset without loading all files.

        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_files': 0,
            'categories': {},
            'category_mapping': self.get_category_mapping()
        }

        category_mapping = self.get_category_mapping()
        target_counts = Counter()

        for original_category, target_category in category_mapping.items():
            if target_category is None:
                continue

            category_path = os.path.join(self.base_path, original_category)

            if os.path.exists(category_path):
                audio_files = glob.glob(os.path.join(category_path, "*.mp3"))
                count = len(audio_files)

                stats['categories'][original_category] = {
                    'count': count,
                    'target_category': target_category
                }

                target_counts[target_category] += count
                stats['total_files'] += count

        stats['target_category_counts'] = dict(target_counts)

        return stats

    def save_processed_audio(
        self,
        audio: np.ndarray,
        output_path: str,
        sr: Optional[int] = None
    ):
        """
        Save processed audio to file.

        Args:
            audio: Audio data
            output_path: Output file path
            sr: Sample rate (uses config if None)
        """
        sr = sr or self.sample_rate
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, audio, sr)
        logger.info(f"Audio saved to {output_path}")
