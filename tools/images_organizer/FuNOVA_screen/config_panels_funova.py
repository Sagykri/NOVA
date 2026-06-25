
"""
config suitable for funova screen batch1 and batch2 - as they have the same panel layout, and the same plate-part gene mapping.
(!) Note that the order of genes in each part differs between batches, so needs to map with distinct functions
"""

import os
import sys
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))

from tools.images_organizer.FuNOVA_screen.config import Config
import pandas as pd

GENES_METADATA_PATH = "/home/projects/hornsteinlab/Collaboration/Guy_Lior/fuNOVA_Screen_B2/virus_metadata.csv"
GENE_NAME_COL = "gene_name"
NUMS_TO_GENES_DICT_PATH = "/home/projects/hornsteinlab/Collaboration/Guy_Lior/fuNOVA_Screen/organizer_files/virus_dict.csv"
GENE_NUM_COL = "virus_number"

CHANNELS_AND_PANNELS_INFO_PATH = "/home/projects/hornsteinlab/Collaboration/Guy_Lior/fuNOVA_Screen_B2/integrated_info.csv"  


parts = [1,2,3]

plate_part_to_col_shift_dict={
    1: 0,
    2: 8,
    3: 16,
}

panel_letter_to_num_dict = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
}

def load_data():
    # virus numbers by real order, plate, part
    df = pd.read_csv(GENES_METADATA_PATH)
    df = df.dropna(subset=[GENE_NUM_COL])
    
    # virus number and gene name mapping
    num_to_name = pd.read_csv(NUMS_TO_GENES_DICT_PATH)
    num_to_name = num_to_name.dropna(subset=[GENE_NUM_COL]).set_index(GENE_NUM_COL).to_dict()[GENE_NAME_COL]

    df['gene_name'] = df[GENE_NUM_COL].map(num_to_name)

    # verify gene_name are not none- 
    if df['gene_name'].isnull().any():
        missing_genes = df[df['gene_name'].isnull()][GENE_NUM_COL].tolist()
        raise ValueError(f"Missing gene names for virus numbers: {missing_genes}")
    
    # for non-unique names add plate suffix ("-pX")
    non_unique_names = df['gene_name'].duplicated(keep=False)
    df.loc[non_unique_names, 'gene_name'] = df.loc[non_unique_names, 'gene_name'].astype(str) + '-p' + df.loc[non_unique_names, 'plate'].astype(str)

    return df

def get_mappings(col_idx_1, col_idx_2, plate):
    df = load_data()
    df_plate = df[df['plate'] == plate]
    
    mapping = {}
    for part in parts:
        col_shift = plate_part_to_col_shift_dict[part]
        genes = df_plate[df_plate['part'] == part][GENE_NAME_COL].tolist()
        for row_idx, gene in enumerate(genes):
            mapping[gene] = [(row_idx+1, col_idx_1 + col_shift), (row_idx+1, col_idx_2 + col_shift)]
    
    return {"C9": mapping}

def get_panel_markers(plate, panel):
    df = pd.read_csv(CHANNELS_AND_PANNELS_INFO_PATH)
    row = df[(df['panel'] == f"Panel{panel}") & (df['plate'] == f"plate{plate}")]
    markers = []
    for ch in ["ch1", "ch2", "ch3", "ch4"]:
        marker = row.get(f'{ch}_marker', None)
        if marker is None:
            raise ValueError(f"Marker for {ch} not found for plate {plate} and panel {panel}")
        markers.append(marker.values[0])
    return markers
# old function for batch1
# def get_mappings(col_idx_1, col_idx_2, plate):
#     # read the metadta file for given plate
#     df = pd.read_csv(GENES_METADATA_PATH)
#     df_plate = df[df['plate'] == plate]
    
#     #remove nan values
#     df_plate = df_plate.dropna(subset=[GENE_NAME_COL])
    
#     unique_parts = df_plate['part'].unique()
#     mapping = {}
#     # for each part, get the genes and map them to their respective row and column indices
#     for part in unique_parts:
#             # each part has a different column shift (1-8, 9-16, 17-24) based on the plate division
#             col_shift = plate_part_to_col_shift_dict[part]

#             # get the genes for the current part and map them to their respective row and column indices
#             genes = df_plate[df_plate['part'] == part][GENE_NAME_COL].tolist()
#             for row_idx, gene in enumerate(genes):
#                 mapping[gene] = [(row_idx+1, col_idx_1 + col_shift), (row_idx+1, col_idx_2 + col_shift)]
    
#     # filter to keep specific genes only (for testing)
#     genes_to_keep = ["non-targeting_00004_00017_p1", "non-targeting_00004_00017_p2",
#                      "non-targeting_00004_00017_p3","non-targeting_00004_00017_p4", 
#                      "non-targeting_00010_00031_p1", "non-targeting_00010_00031_p2", "non-targeting_00010_00031_p3", "non-targeting_00010_00031_p4","non-targeting_00035_00050_p1", "non-targeting_00035_00050_p2", "non-targeting_00035_00050_p3", "non-targeting_00035_00050_p4", "non-targeting_00053_00059_p1", "non-targeting_00053_00059_p2", "non-targeting_00053_00059_p3", "non-targeting_00053_00059_p4", "non-targeting_00111_00121_p1", "non-targeting_00111_00121_p2", "non-targeting_00111_00121_p3", "non-targeting_00111_00121_p4", 
#                      "Empty_p1", "Empty_p2", "Empty_p3", "Empty_p4",
#                      "TDP-43_p3", "TDP-43_p4", 
#                      "Ranbp17_p3", "Ranbp17_p4"]
#     mapping = {gene: mapping[gene] for gene in genes_to_keep if gene in mapping}
    
#     return {"C9": mapping}

class Config_Base_Data(Config):
    def __init__(self, batch):
        super().__init__()

        self.BATCH = batch
        self.TOTAL_NUM_COLUMNS = 24
        self.THIRD_PLATE_NUM_COLUMNS = self.TOTAL_NUM_COLUMNS // 3

class Config_Base_4Markers(Config_Base_Data):
    def __init__(self, batch, plate):
        super().__init__(batch)
        self.PLATE = plate
        self.FOLDERS = [f"plate{self.PLATE}"]
        self.panel_number = panel_letter_to_num_dict[self.panel]
        self.INCLUDE_SUB_FOLDERS = [f'plate{self.PLATE}/Panel{self.panel_number}']
        self.CONFIG = {
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3", "ch4"],
            self.KEY_REPS: ["rep1", "rep2"],
            self.KEY_CELL_LINES: get_mappings(self.col_rep1, self.col_rep2, self.PLATE),
            self.KEY_MARKERS: {f'Panel{self.panel_number}': self.markers}
        }


class Config_A(Config_Base_4Markers):
    def __init__(self, batch, plate):
        
        # Params:
        self.col_rep1 = 1
        self.col_rep2 = 2
        self.panel = 'A'
        self.markers = get_panel_markers(plate, self.panel)
        
        super().__init__(batch, plate)

class Config_B(Config_Base_4Markers):
    def __init__(self, batch, plate):

        # Params:
        self.col_rep1 = 3
        self.col_rep2 = 4
        self.panel = 'B'
        self.markers = get_panel_markers(plate, self.panel)

        super().__init__(batch, plate)

class Config_C(Config_Base_4Markers):
    def __init__(self, batch, plate):

        # Params:
        self.col_rep1 = 5
        self.col_rep2 = 6
        self.panel = 'C'
        self.markers = get_panel_markers(plate, self.panel)

        super().__init__(batch, plate)

class Config_D(Config_Base_4Markers):
    def __init__(self, batch, plate):

        # Params:
        self.col_rep1 = 7
        self.col_rep2 = 8
        self.panel = 'D'
        self.markers = get_panel_markers(plate, self.panel)

        super().__init__(batch, plate)
        panel_number = panel_letter_to_num_dict[self.panel]
        self.INCLUDE_SUB_FOLDERS = [f'plate{self.PLATE}/Panel{panel_number}A', f'plate{self.PLATE}/Panel{panel_number}B']


if __name__ == "__main__":
    get_panel_markers(4, 2)
    


