import numpy as np
import os
import sys

sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")
from src.common.utils import load_config_file
from src.embeddings.embeddings_utils import load_embeddings
from utils import compute_label_pair_distances_stats

def generate_distances(
    embeddings_folder: str,
    config_path_data: str,
    metric: str = "euclidean",
    detailed_stats: bool = False,
    folder_name: str = "output"
):
    # Load config and embeddings
    config_data = load_config_file(config_path_data, 'data')
    config_data.OUTPUTS_FOLDER = embeddings_folder
    embeddings, labels, _ = load_embeddings(embeddings_folder, config_data)
    print(
            f"Loaded {len(embeddings)} embeddings "
            f"with {len(np.unique(labels))} unique labels."
        )

    # Compute stats
    df_stats = compute_label_pair_distances_stats(
        embeddings=embeddings,
        labels=labels,
        metric=metric,
        full_stats=detailed_stats  
    )
    # Save results
    config_name = os.path.basename(config_path_data)
    output_dir = os.path.join(os.path.dirname(__file__), folder_name)
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(
        output_dir,
        f"label_pair_distances_stats_{config_name}_{metric}_detaild:{detailed_stats}_gpu.csv"
    )
    df_stats.to_csv(out_csv, index=False)
    print(f"Saved distance stats to {out_csv}")

if __name__ == "__main__":
    config_path_data  = sys.argv[1]
    embeddings_folder = sys.argv[2]
    print(f"Embeddings folder: {embeddings_folder}")
    print(f"Config path: {config_path_data}")
    detailed_stats = True  # Set to True for detailed statistics
    metric = "euclidean"
    output_folder = "output_distances"

    generate_distances(
            embeddings_folder=embeddings_folder,
            config_path_data=config_path_data,
            metric=metric,
            detailed_stats=detailed_stats,
            folder_name=output_folder
        )
    print("Analysis complete.")
