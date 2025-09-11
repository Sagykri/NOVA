import logging
import os
from typing import Dict, List, Tuple
import numpy as np
import sys

sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.datasets.dataset_base import DatasetBase
from src.datasets.dataset_NOVA import DatasetNOVA
from src.datasets.dataset_config import DatasetConfig


class DatasetFromPaths(DatasetNOVA):
    """
    Dataset class that loads data from a given list of image paths instead of scanning folders.
    Inherits from DatasetNOVA and overrides only the loading logic.
    """

    def __init__(self, dataset_config: DatasetConfig, image_paths: List[str]):
        """
        Args:
            dataset_config (DatasetConfig): Dataset configuration.
            image_paths (List[str]): List of full paths to .npy image files.
        """
        self.image_paths = image_paths
        super().__init__(dataset_config)

    def _load_data_paths(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads image paths and generates labels from a predefined list of paths.
        Applies filters and labeling consistent with DatasetNOVA.
        """
        image_paths = []
        labels = []

        for path in self.image_paths:
            if not path.endswith(".npy"):
                logging.info(f"Skipping unsupported file type: {path}")
                continue

            marker_folder = os.path.dirname(path)

            if not self._DatasetNOVA__passes_filters(marker_folder):
                continue

            rep = self._DatasetNOVA__extract_rep(os.path.basename(path))
            if self._DatasetNOVA__get_filter_criteria()['reps'] and rep not in self._DatasetNOVA__get_filter_criteria()['reps']:
                logging.info(f"Skipping rep {rep} (not in rep filter list).")
                continue

            label = self._DatasetNOVA__generate_label(marker_folder, rep)
            image_paths.append(path)
            labels.append(label)

        image_paths = np.array(image_paths)
        labels = np.array(labels).reshape(-1, 1)

        logging.info(f"{len(image_paths)} files processed from explicit list, {labels.shape[0]} labels generated")

        return image_paths, labels
