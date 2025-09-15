import re
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
from src.datasets.dataset_config import DatasetConfig


def load_labels_from_npy(embd_dir, set_type):
    labels_path = os.path.join(embd_dir, f"{set_type}_labels.npy")
    labels = np.load(labels_path, allow_pickle=True)
    labels_df = pd.DataFrame(labels, columns=['full_label'])
    labels_df[['protein', 'condition', 'treatment', 'batch', 'replicate']] = labels_df['full_label'].str.split('_', expand=True)

    return labels_df

def display_labels(df:pd.DataFrame, save_dir: str = None):
    grouped = df.groupby(['protein', 'condition', 'treatment'])
    print("labels_df:")
    print(df.shape)
    print(df.head())

    print("\nlabels groups:")
    print(grouped.size())

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"labels.csv")
        df.to_csv(save_path, index=False)

def load_npy_to_df(input_dir, file_name, columns= None):
    path = os.path.join(input_dir, file_name)
    data = np.load(path, allow_pickle=True)
    df = pd.DataFrame(data, columns = columns)
    return df

def load_npy_to_nparray(input_dir, file_name):
    path = os.path.join(input_dir, file_name)
    data = np.load(path, allow_pickle=True)
    return data


def load_embeddings_from_npy(embd_dir, set_type):
    embeddings_path = os.path.join(embd_dir, f"{set_type}.npy")
    embeddings = np.load(embeddings_path, allow_pickle=True)
    embeddings_df = pd.DataFrame(embeddings)
    return embeddings_df

def load_attn_maps_from_npy(embd_dir, set_type):
    attn_path = os.path.join(embd_dir, f"{set_type}_attn.npy")
    attn = np.load(attn_path, allow_pickle=True)
    attn_df = pd.DataFrame(attn)
    return attn_df

def display_embeddings(df:pd.DataFrame, save_dir: str = None):
    print("embeddings_df:")
    print(df.shape)
    print(df.head())

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"embeddings.csv")
        df.to_csv(save_path, index=False)


def load_paths_from_npy(input_path, set_type = None):

    # Load data
    if set_type is not None:
        paths = np.load(os.path.join(input_path, f"{set_type}_paths.npy"), allow_pickle=True)
    else:
        paths = np.load(input_path, allow_pickle=True)
    df = parse_paths(paths)

    return df

def parse_paths(paths):
    """
    args:
        paths:  list/array of full path of format: 
                <path>/batch{X}/{Cell_Line}/{Condition}/{Marker}/rep{X}_R11_w2confmCherry_s{X}_panel{X}_SCNA_processed.npy/{TILE}
    
    returns:
        df: parsed df woth columns: ["Batch", "Condition", "Rep", "Site", "Panel", "Cell_Line", "Tile", "Path"]
    """

    # Regex pattern to extract Batch, Condition, Rep, Raw Image Name, Panel, Cell Line, and Tile
    # pattern = re.compile(
    # r".*/[Bb]atch(\d+)/([^/]+)/([^/]+)/([^/]+)/(rep\d+)_.*_(s\d+)_?(panel\w+)_.*_processed\.npy/(\d+)"
    # )
    # pattern = re.compile(
    # r".*/[Bb]atch(\d+)/([^/]+)/([^/]+)/([^/]+)/(rep\d+)_.*?(?:f(\d+)[^/]*|_s(\d+))_(panel\w+)_.*_processed\.npy/(\d+)"
    # )


    # # Parsing the paths
    # parsed_data = [pattern.match(path).groups() for path in paths if pattern.match(path)]

    # if len(parsed_data) != len(paths):
    #     raise RuntimeError("in parse_paths: not all paths match the regex pattern.")
    # # Convert metadata to DataFrame
    # df = pd.DataFrame(parsed_data, columns=[
    # "Batch", "Cell_Line", "Condition", "Marker", "Rep", "Site", "Panel", "Tile"
    # ])

    # Regex with named capture groups
    pattern = re.compile(
        r".*/[Bb]atch(?P<Batch>\d+)/(?P<Cell_Line>[^/]+)/(?P<Condition>[^/]+)/(?P<Marker>[^/]+)/"
        r"(?P<Rep>rep\d+)_.*?(?:f(?P<Site_f>\d+)|s(?P<Site_s>\d+)).*?_"
        r"(?P<Panel>panel\w+)_.*_processed\.npy/(?P<Tile>\d+)"
    )
    parsed_data = []
    for path in paths:
        match = pattern.match(path)
        if not match:
            raise RuntimeError(f"in parse_paths: path did not match pattern: {path}")
        data = match.groupdict()
        data["Site"] = data["Site_f"] or data["Site_s"]  # Normalize site
        del data["Site_f"]
        del data["Site_s"]
        parsed_data.append(data)

    # Convert to DataFrame
    df = pd.DataFrame(parsed_data)
    df['Path'] = paths
    df['File_Name'] = [os.path.basename(path.split('.npy')[0]) for path in paths]

    return df

def parse_path_item(path_item):
    img_path = str(path_item.Path).split('.npy')[0]+'.npy'
    tile = int(path_item.Tile)
    Site = path_item.Site
    return img_path, tile, Site



def display_paths(df:pd.DataFrame, save_dir: str = None):
    grouped = paths_df.groupby(['Cell_Line', 'Condition', 'Site', 'Rep'])
    print("paths_df:")
    print(df.shape)
    print(df.head())
    print("\npaths groups:")
    print(grouped.size())

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"paths.csv")
        df.to_csv(save_path, index=False)


def load_tile(path, tile):
    """
    args:
        path:   path of the original img. should be of size (num_tiles, H, W, num_ch)

    returns:
        marker:     normalized img matrix for the marker (ch0)
        nucleus:    normalized img matrix for the nucleus (ch1)
        overlay:    overlay of the marker on top of the nucleus (Red for marker, Green for nucleus)
    """
    # Load the image
    image = np.load(path)
    site_image = image[tile]
    marker = site_image[:, :, 0]
    nucleus = site_image[:, :, 1]

    # Normalize
    marker1 = (marker - marker.min()) / (marker.max() - marker.min())
    nucleus1 = (nucleus - nucleus.min()) / (nucleus.max() - nucleus.min())

    # Create RGB overlay: 
    overlay = np.zeros((*marker.shape, 3)) # black background
    overlay[..., 2] = nucleus      # blue channel = nucleus
    overlay[..., 1] = marker     # Green channel = marker

    return marker, nucleus, overlay

def display_tile(Site:str, tile:int, marker:np.array, nucleus:np.array, overlay:np.array, save_dir:str = None):

    # Plot target, nucleus, and overlay
    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    ax[0].set_title(f'{Site}/{tile} - Marker', fontsize=11)
    ax[0].imshow(marker, cmap='gray', vmin=0, vmax=1)
    ax[0].set_axis_off()
    ax[1].set_title(f'{Site}/{tile} - Nucleus', fontsize=11)
    ax[1].imshow(nucleus, cmap='gray', vmin=0, vmax=1)
    ax[1].set_axis_off()
    ax[2].set_title(f'{Site}/{tile} - Overlay', fontsize=11)
    ax[2].imshow(overlay)
    ax[2].set_axis_off()


    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{Site}_tile{tile}.png")
        print(f"saving fig on {save_path}")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        plt.show()

def __extract_indices_to_plot(keep_samples_dirs: list[str], paths: np.ndarray, data_config: DatasetConfig, use_settype = True):
    """
    Extract indices to plot from a list of keep_samples_dirs.
    For each dataset split (train/val/test or test), collects indices from all directories and concatenates them.

    Parameters:
        keep_samples_dirs: list of directories containing .npy files of sample paths
        paths: np.ndarray of path arrays, one per dataset split
        data_config: dataset configuration object

    Returns:
        all_samples_indices: list of lists, where each sublist contains indices for one dataset split
    """
    if data_config.SPLIT_DATA:
        data_set_types = ['trainset', 'valset', 'testset']
    else:
        data_set_types = ['testset']

    all_samples_indices = []

    for i, set_type in enumerate(data_set_types):
        cur_paths = paths[i]
        paths_df = parse_paths(cur_paths)

        # Accumulate all keep_paths from all dirs
        combined_keep_paths = set()
        for dir_path in keep_samples_dirs:
            if use_settype:
                keep_paths_df = load_paths_from_npy(dir_path, set_type)
            else:
                keep_paths_df = load_paths_from_npy(dir_path)
            combined_keep_paths.update(keep_paths_df["Path"].tolist())

        # Get indices of matching paths
        samples_indices = paths_df[paths_df["Path"].isin(combined_keep_paths)].index.tolist()
        all_samples_indices.append(samples_indices)

    return all_samples_indices

def __extract_samples_to_plot(sampels: np.ndarray[str], indices:list, data_config: DatasetConfig):
    """
    extract samples from given array using indices. 
    """
    if data_config.SPLIT_DATA:
        data_set_types = ['trainset','valset','testset']
    else:
        data_set_types = ['testset']
    
    all_filtered_sampels = []
    for i, set_type in enumerate(data_set_types):
        curr_samples, curr_indices = sampels[i], indices[i]
        filtered_samples = curr_samples[curr_indices]
        all_filtered_sampels.append(filtered_samples)
        
    return all_filtered_sampels

 


if __name__ == "__main__":
    # input_dir = "/home/projects/hornsteinlab/giliwo/NOVA_rotation/attention_maps/attention_maps_output/RotationDatasetConfig_Euc_Pairs_all_layers/raw/attn_maps/neurons/batch9"
    # emb_df = load_paths_from_npy(input_dir, "testset")
    # print(np.array(emb_df[emb_df["File_Name"] == "rep1_R11_w3confCy5_s26_panelA_WT_processed"].Path))

    # img_path = "/home/projects/hornsteinlab/Collaboration/MOmaps/input/images/processed/spd2/SpinningDisk/batch9/WT/stress/G3BP1/rep1_R11_w3confCy5_s60_panelA_WT_processed.npy"
    # load_tile(img_path, 1)

    path = "/home/projects/hornsteinlab/giliwo/NOVA_rotation/attention_maps/attention_maps_output/finetuned_model_old/raw/neurons/batch9"
    emb = load_embeddings_from_npy(path, "testset")
    print(emb.shape)
