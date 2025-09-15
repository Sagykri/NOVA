import os
import sys

sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

import logging
from src.common.utils import load_config_file
from src.embeddings.embeddings_utils import load_embeddings
from src.datasets.dataset_config import DatasetConfig
from src.analysis.analyzer_attention_scores import AnalyzerAttnScore
from src.figures.attn_scores_plotting import plot_corr_data
from src.figures.plot_attn_score_config import PlotAttnScoreConfig

def load_attn_and_plot_correlation(outputs_folder_path:str, config_path_data:str, 
                                   config_path_corr:str, config_path_plot:str = None):

    config_data:DatasetConfig = load_config_file(config_path_data, "data")
    config_data.OUTPUTS_FOLDER = outputs_folder_path
    config_corr = load_config_file(config_path_corr, "data")

    # load processed attn maps
    processed_attn_maps, labels, paths = load_embeddings(os.path.join(outputs_folder_path, "attention_maps"), config_data, emb_folder_name = "processed")
    processed_attn_maps, labels, paths = [processed_attn_maps], [labels], [paths] #TODO: fix, needed for settypes
    
    logging.info("[Generate correlations]")
    d = AnalyzerAttnScore(config_data, output_folder_path, config_corr)
    corr_data = d.calculate(processed_attn_maps, labels, paths)
    d.save()

    # save summary plots of the correlations
    if config_path_plot is not None:
        config_plot:PlotAttnScoreConfig = load_config_file(config_path_plot, 'plot')

        if config_plot.PLOT_CORR_SUMMARY:
            plot_corr_data(corr_data, labels, config_data, config_plot, config_corr.CORR_METHOD, output_folder_path=d.get_saving_folder(feature_type='attn_scores'), features_names=config_corr.FEATURES_NAMES)

        

if __name__ == "__main__":
    print("Starting generating distances...")
    try:
        if len(sys.argv) < 4:
            raise ValueError("Invalid arguments. Must supply output folder path and data config!")
        output_folder_path = sys.argv[1]
        config_path_data = sys.argv[2]
        config_path_corr = sys.argv[3]

        if len(sys.argv) == 5:
            config_path_plot = sys.argv[4]
        else:
            config_path_plot = None
        load_attn_and_plot_correlation(output_folder_path, config_path_data, config_path_corr, config_path_plot)

    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
