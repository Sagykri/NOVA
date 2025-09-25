import pandas as pd
import logging
import sys
import os
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.classifier.classifier_config import ClassifierConfig
from src.analysis.analyzer import Analyzer
from src.analysis.analyzer_classification_utils import (
    run_baseline_model,
    run_train_test_split_baseline,
)


class AnalyzerClassification(Analyzer):
    """Analyzer class is used to train/evaluate classifiers.
    The classification analyzer's three main methods are:
    1. calculate(): to run the classifier(s) and collect predictions/scores.
    2. load(): to load previously saved classification outputs.
    3. save(): to save classification outputs.
    """

    def __init__(self, classification_config: ClassifierConfig, output_folder_path: str):
        """Get an instance

        Args:
            classification_config (ClassifierConfig): The classifier config object.
            output_folder_path (str): path to output folder.
        """
        self.__set_params(classification_config, output_folder_path)
        self.save_path = self.__get_save_dir()

    def __set_params(self, classification_config: ClassifierConfig, output_folder_path: str) -> None:
        """Extracting params from the configuration

        Args:
            classification_config (ClassifierConfig): classification configuration
            output_folder_path (str): path to output folder
        """
        self.classification_config = classification_config
        self.output_folder_path = output_folder_path

    # ---------- core API ----------
    def calculate_and_plot(self):
        """
        Calculate classification outputs and store results.
        Returns:
            The raw return from the underlying runner.
        """
        cfg = self.classification_config
        logging.info(f"[Calculate classification] classification_config: {type(cfg)}, output_folder_path: {self.output_folder_path}, classifier: {cfg.classifier}, mode: {cfg.mode}")

        common = dict(
            dataset_config=cfg.dataset_config(),
            batches=cfg.batches,
            balance=cfg.balance,
            norm=cfg.norm,
            choose_features=cfg.choose_features,
            top_k=cfg.top_k,
            apply_pca=cfg.apply_pca,
            pca_components=cfg.pca_components,
            classifier_class=cfg.classifier_class(),
            classifier_kwargs=cfg.classifier_kwargs or {},
            get_proba=cfg.get_proba,
            label_map=cfg.label_map,
            save_path=self.save_path,
        )

        if cfg.mode == "singleton":
            out = run_baseline_model(
                **common,
                train_each_as_singleton=True,
                test_specific_batches=cfg.test_specific_batches,
                train_specific_batches=cfg.train_specific_batches,
                results_csv=cfg.results_csv,
                plot_per_fold_cm=cfg.plot_per_fold_cm,
            )
        elif cfg.mode == "loocv":
            out = run_baseline_model(
                **common,
                train_each_as_singleton=False,
                test_specific_batches=cfg.test_specific_batches,
                train_specific_batches=cfg.train_specific_batches,
                results_csv=cfg.results_csv,
                plot_per_fold_cm=cfg.plot_per_fold_cm,
            )
        elif cfg.mode == "train_test_split":
            out = run_train_test_split_baseline(**common)
        else:
            raise ValueError("mode must be one of {'singleton','loocv','train_test_split'}")
        self.save(out)

    def __get_save_dir(self) -> str:
        """
        Return folder path for this run (params encoded in folder name).
        """

        model_output_folder = self.output_folder_path
        base = os.path.join(model_output_folder, 'classifications')
        os.makedirs(base, exist_ok=True)

        cfg = self.classification_config
        parts = [
            cfg.classifier,
            cfg.mode,
            ("B" + "-".join(map(str, cfg.batches))) if cfg.batches else "Bnone",
            ("trainB-" + "-".join(map(str, cfg.train_specific_batches))) if cfg.train_specific_batches else None,
            ("testB-" + "-".join(map(str, cfg.test_specific_batches))) if cfg.test_specific_batches else None,
            f"bal{int(cfg.balance)}",
            f"norm{int(cfg.norm)}",
            (f"fs{cfg.top_k}" if cfg.choose_features else "fs0"),
            (f"pca{cfg.pca_components}" if cfg.apply_pca else "pca0"),
        ]
        folder = "_".join([p for p in parts if p])
        out_dir = os.path.join(base, folder)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir
    
    def save(self, out):
        """
        Save the calculated classification results (macro_avg) to a CSV file.
        """
        if self.save_path is not None:
            savepath = os.path.join(self.save_path, "classification_results.csv")
        
            pd.DataFrame([out]).to_csv(savepath, index=False)

            logging.info(f"Saved classification results to {savepath}")
        else:
            logging.warning("Save path is None, cannot save classification results.")