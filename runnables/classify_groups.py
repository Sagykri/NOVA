import os
import sys

sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

import logging
from src.common.utils import load_config_file
from src.classifier.classifier_config import ClassifierConfig
from src.analysis.analyzer_classification import AnalyzerClassification


def classify_groups(output_folder_path: str, config_class_path: str) -> None:
    logging.info("[classify_groups] start")

    classification_config:ClassifierConfig = load_config_file(config_class_path, 'data')
    classification_config.OUTPUTS_FOLDER = output_folder_path

    d = AnalyzerClassification(classification_config, output_folder_path)
    d.calculate_and_plot()

if __name__ == "__main__":
    print("Starting classify_groups...")
    try:
        if len(sys.argv) < 3:
            raise ValueError(
                "Invalid arguments. Must supply output folder path and config class path "
                "(e.g., 'manuscript.classification_config.NIHUMAP1LinearSVCConfig')."
            )
        output_folder_path = sys.argv[1]
        config_class_path = sys.argv[2]
        classify_groups(output_folder_path, config_class_path)
    except Exception as e:
        logging.exception(str(e))
        raise
    logging.info("[classify_groups] Done")
