import os
import sys

sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

import logging
from src.common.utils import load_config_file
from src.classifier.classifier_config import ClassifierConfig
from src.figures.plot_config import PlotConfig
from src.analysis.analyzer_classification import AnalyzerClassification


def classify_groups(output_folder_path: str, config_class_path: str, config_path_plot: str) -> None:
    logging.info("[classify_groups] start")

    classification_config:ClassifierConfig = load_config_file(config_class_path, 'data')
    classification_config.OUTPUTS_FOLDER = output_folder_path

    if config_path_plot is not None:
        config_plot:PlotConfig = load_config_file(config_path_plot, 'plot')
        color_mappings = config_plot.COLOR_MAPPINGS  
        # build label_map 
        label_map = {k: v["alias"] for k, v in color_mappings.items()}
        classification_config.label_map = label_map
        logging.info(f"Using label_map from plot config")

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
        if len(sys.argv) > 3: 
            # optional plot config for label mapping
            config_path_plot = sys.argv[3]
        else:
            config_path_plot = None
        classify_groups(output_folder_path, config_class_path, config_path_plot)
    except Exception as e:
        logging.exception(str(e))
        raise
    logging.info("[classify_groups] Done")
