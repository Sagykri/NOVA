import os
import sys


sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

import logging

from src.common.utils import load_config_file, get_if_exists
from src.embeddings.embeddings_utils import load_embeddings
from src.figures.umap_plotting import plot_umap
from src.datasets.dataset_config import DatasetConfig
from src.figures.plot_config import PlotConfig
from src.analysis.analyzer_umap_single_markers import AnalyzerUMAPSingleMarkers
from src.analysis.analyzer_umap_multiple_markers import AnalyzerUMAPMultipleMarkers
from src.analysis.analyzer_umap_multiplex_markers import AnalyzerUMAPMultiplexMarkers
from src.analysis.analyzer_umap import AnalyzerUMAP

# Mapping between umap_type and corresponding Analyzer classes and plotting functions
analyzer_mapping = {
    0: (AnalyzerUMAPSingleMarkers, AnalyzerUMAP.UMAPType(0).name),
    1: (AnalyzerUMAPMultipleMarkers, AnalyzerUMAP.UMAPType(1).name),
    2: (AnalyzerUMAPMultiplexMarkers, AnalyzerUMAP.UMAPType(2).name)
}

def generate_umaps(output_folder_path:str, config_path_data:str, config_path_plot:str)->None:
    config_data:DatasetConfig = load_config_file(config_path_data, 'data')
    config_data.OUTPUTS_FOLDER = output_folder_path
    config_plot:PlotConfig = load_config_file(config_path_plot, 'plot')

    embeddings, labels, paths = load_embeddings(output_folder_path, config_data)
    umap_idx = get_if_exists(config_plot, 'UMAP_TYPE', None)
    if umap_idx not in analyzer_mapping:
        raise ValueError(f"Invalid UMAP index: {umap_idx}. Must be one of {list(analyzer_mapping.keys())}, and defined in plot config.")
    
    AnalyzerUMAPClass, UMAP_name = analyzer_mapping[umap_idx]
    logging.info(f"[Generate {UMAP_name} UMAP]")

    # Create the analyzer instance
    analyzer_UMAP:AnalyzerUMAP = AnalyzerUMAPClass(config_data, output_folder_path)
    
    # Define the output folder path
    saveroot = analyzer_UMAP.get_saving_folder(feature_type = os.path.join('UMAPs', analyzer_UMAP.UMAPType(umap_idx).name))  
    colored_by = get_if_exists(config_plot, 'MAP_LABELS_FUNCTION',None)
    if colored_by is not None:
        saveroot += f'_colored_by_{colored_by}'
    to_color = get_if_exists(config_plot, 'TO_COLOR',None)
    if to_color is not None:
        saveroot += f'_coloring_{to_color[0].split("_")[0]}'

    os.makedirs(saveroot, exist_ok=True)
    logging.info(f'saveroot: {saveroot}')
    
    # Calculate the UMAP embeddings
    umap_embeddings, labels, paths, ari_scores = analyzer_UMAP.calculate(embeddings, labels, paths)
    # Plot the UMAP
    plot_umap(umap_embeddings, labels, config_data, config_plot, saveroot, umap_idx, ari_scores, paths)
                

if __name__ == "__main__":
    print("Starting generating umaps...")
    try:
        if len(sys.argv) < 4:
            raise ValueError("Invalid arguments. Must supply output folder path, data config and plot config.")
        output_folder_path = sys.argv[1]
        config_path_data = sys.argv[2]
        config_path_plot = sys.argv[3]

        generate_umaps(output_folder_path, config_path_data, config_path_plot)

    except Exception as e:
        logging.exception(str(e))
        raise e
    logging.info("Done")
