import os
import sys
import numpy as np
import pandas as pd
import time

sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

from src.common.utils import load_config_file
from src.embeddings.embeddings_utils import load_embeddings
from utils import compute_label_pair_distances_stats, get_base_to_reps
from src.analysis.analyzer_multiplex_markers import AnalyzerMultiplexMarkers

def parse_args(argv):
    """
    Parse arguments.

    Args:
        argv (List[str]): List of command-line arguments.

    Returns:
        dict: Parsed values.
    """
    if len(argv) < 3:
        raise ValueError("Usage: caclculate_distances.py <config_path_data> <embeddings_folder> [multiplexed] [detailed_stats] ([] optional)")

    return {
        'config_path_data' : sys.argv[1],
        'embeddings_folder' : sys.argv[2],
        'multiplexed': True if "multiplexed" in sys.argv else False,
        'detailed_stats': True if "detailed" in sys.argv else False,
    }

def generate_distances(
    embeddings_folder: str,
    config_path_data: str,
    metric: str = "euclidean",
    detailed_stats: bool = False,
    folder_name: str = "output",
    multiplexed: bool = False,
):
    # Load config and embeddings
    config_data = load_config_file(config_path_data, 'data')
    config_data.OUTPUTS_FOLDER = embeddings_folder
    config_data.ADD_REP_TO_LABEL = True ## Force adding rep to label for distance calculation

    embeddings, labels, _ = load_embeddings(embeddings_folder, config_data)

    if multiplexed:
        print("Multiplexed embeddings detected, transforming embeddings and labels using AnalyzerMultiplexMarkers.")
        analyzer_multiplex = AnalyzerMultiplexMarkers(config_data, embeddings_folder)
        embeddings, labels, _ = analyzer_multiplex.calculate(embeddings, labels)  
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) 

    print(f"Loaded {len(embeddings)} embeddings "
            f"with {len(np.unique(labels))} unique labels.")
    print(f"example label: {labels[0] if len(labels) > 0 else 'None'}")

    # Compute stats
    base_to_reps = get_base_to_reps(labels)
    print(f"Found {len(base_to_reps)} base labels with their reps.")
    print(f"Example base label: {list(base_to_reps.keys())[0]} with reps {base_to_reps[list(base_to_reps.keys())[0]]}")
    all_dfs = []
    # Iterate over base labels and their reps
    for base, reps in base_to_reps.items():
        if len(reps) < 2:
            print(f"Skipping base label '{base}' with reps {reps} (less than 2 reps)")
            continue
        mask = np.isin(labels, reps)
        filtered_indices = np.where(mask)[0]
        embeddingsi = embeddings[filtered_indices]
        labelsi = labels[filtered_indices]

        df_part = compute_label_pair_distances_stats(
            embeddings=embeddingsi,
            labels=labelsi,
            metric=metric,
            full_stats=detailed_stats
        )
        all_dfs.append(df_part)

    df_stats = pd.concat(all_dfs, ignore_index=True)
    
    # Save results
    config_name = os.path.basename(config_path_data)
    output_dir = os.path.join(os.path.dirname(__file__), folder_name)
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(
        output_dir,
        f"rep_distances_stats_{config_name}_{metric}_detailed:{detailed_stats}.csv"
    )
    df_stats.to_csv(out_csv, index=False)
    print(f"Saved distance stats to {out_csv}")

if __name__ == "__main__":
    print("Starting calculating distances...")
    try:
        args = parse_args(sys.argv)

        config_path_data = args['config_path_data']
        embeddings_folder = args['embeddings_folder']
        multiplexed = args['multiplexed'] # optional flag: True if "multiplexed" in sys.argv else False
        detailed_stats = args['detailed_stats'] # optional flag: True if "detailed" in sys.argv else False
        metric = "euclidean"  # Default metric

        print(f"Parameters: data config path:{config_path_data}, \
            Embeddings folder:{embeddings_folder}, Multiplexed:{multiplexed}, Detailed stats:{detailed_stats}")

        generate_distances(
                embeddings_folder=embeddings_folder,
                config_path_data=config_path_data,
                metric=metric,
                detailed_stats=detailed_stats,
                folder_name="output_distances",
                multiplexed=multiplexed
            )
        print("Distance calculation completed.")

    except Exception as e:
        print(str(e))
        raise e
    print("Done")
