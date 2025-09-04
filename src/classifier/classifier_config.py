from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --- sklearn / cuML imports ---
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from cuml.linear_model import LogisticRegression as cuMLLogReg
from cuml.ensemble import RandomForestClassifier as cuRF

from src.embeddings.embeddings_config import EmbeddingsConfig

# ---- registry ----
CLASSIFIERS = {
    "LinearSVC": LinearSVC,                               # fast linear SVM (no predict_proba)
    "LogisticRegression": LogisticRegression,             # CPU; supports predict_proba
    "RidgeClassifier": RidgeClassifier,                   # linear model with L2 penalty
    "GaussianNB": GaussianNB,                             # naive Bayes (works well with few features)
    "ExtraTreesClassifier": ExtraTreesClassifier,         # CPU extremely randomized trees
    "cuMLLogisticRegression": cuMLLogReg,                 # GPU LogisticRegression
    "cuMLRandomForest": cuRF,                             # GPU RandomForest
}

class ClassifierConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()
        # ---- data (what/where to load) ----
        self.OUTPUTS_FOLDER: str = ""            # directory containing model output files
        self.multiplexed: bool = False           # if embeddings should be multiplexed
        self.config_fmt: str = ""                # per-batch config name format, e.g. "NIH_..._B{batch}"
        self.config_file: str = ""               # config file containing per-batch configs

        # ---- experiment (which samples/labels) ----
        self.batches: List[int] = [1, 2, 3]   # batch IDs to include
        self.label_map: Optional[Dict[Any, Any]] = None       # remap/merge labels, e.g. {"WT":0,"C9":1}
        self.remove_untreated: bool = False    # whether to remove 'untreated' from labels strings

        # ---- features (preprocessing/selection) ----
        self.balance: bool = False               # rebalance classes during training (e.g., oversampling)
        self.norm: bool = False                  # standardize features (mean=0, std=1)
        self.choose_features: bool = False       # enable univariate feature selection
        self.top_k: int = 100                    # number of features to keep if choose_features=True
        self.apply_pca: bool = False             # enable PCA dimensionality reduction
        self.pca_components: int = 50            # #components if apply_pca=True

        # ---- model (estimator choice + its kwargs) ----
        self.classifier: str = "LinearSVC"       # key in CLASSIFIERS
        self.classifier_kwargs: Dict[str, Any] = {}  # passed to estimator ctor

        # ---- cv / run mode (how to split/train/test) ----
        self.mode: str = "singleton"             # "singleton" (train_each_as_singleton), "loocv", "train_test_split"
        self.test_specific_batches: Optional[List[int]] = None  # explicit test batches (else default per mode)
        self.train_specific_batches: Optional[List[int]] = None # explicit train batches (optional)
        self.get_proba: bool = True          # calculate class probabilities / scores (if supported by classifier) for ROC-AUC
        self.plot_per_fold_cm: bool = True       # draw confusion matrix per fold

        # ---- output (artifacts) ----
        self.results_csv: Optional[str] = None   # path to append results rows (or None)

    # ---- helpers ----
    def classifier_class(self):
        cls = CLASSIFIERS.get(self.classifier)
        if cls is None:
            raise ValueError(f"Unknown/unavailable classifier: {self.classifier}")
        return cls

    def dataset_config(self) -> Dict[str, Any]:
        return {
            # keep your external key expected by the pipeline:
            "path_to_embeddings": self.OUTPUTS_FOLDER,
            "multiplexed": self.multiplexed,
            "config_fmt": self.config_fmt,
            "config_file": self.config_file,
        }

    def __post_init__(self):
        if self.mode not in {"singleton", "loocv", "train_test_split"}:
            raise ValueError(f"mode invalid: {self.mode}")