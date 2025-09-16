from src.classifier.classifier_config import ClassifierConfig

# ---- NIH config ----
class NIHUMAP1LinearSVCConfig(ClassifierConfig):
    def __init__(self):
        super().__init__()  # keep Base defaults
        # ---- data to load ----
        self.multiplexed = False
        self.config_fmt = "NIH_UMAP1_DatasetConfig_B{batch}"   # per-batch config name format
        self.config_file = "manuscript/manuscript_figures_data_config"  # config file containing per-batch configs

        self.batches = [1, 2, 3]          # batch IDs to include

        # ---- model (estimator choice + its kwargs) ----
        self.classifier = "LinearSVC"      # key in CLASSIFIERS
        self.classifier_kwargs = {"C": 1.0, "max_iter": 1000, "random_state": 42}
        self.mode = "singleton"            # "singleton" (train_each_as_singleton)

        # ---- output (artifacts) ----
        # self.results_csv = "classification_results.csv"

