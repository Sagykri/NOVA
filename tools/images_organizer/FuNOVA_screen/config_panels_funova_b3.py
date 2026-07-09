
"""
config suitable for funova screen batch3 - as it has distinct panel layout compared to batch1+2

"""

import os
import sys
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))

from tools.images_organizer.FuNOVA_screen.config import Config
import pandas as pd

BATCH = 3
GENES_METADATA_PATH = f"/home/projects/hornsteinlab/Collaboration/Guy_Lior/fuNOVA_Screen_B{BATCH}/virus_metadata.csv"
GENE_NAME_COL = "gene_name"
NUMS_TO_GENES_DICT_PATH = "/home/projects/hornsteinlab/Collaboration/Guy_Lior/fuNOVA_Screen/organizer_files/virus_dict.csv"
GENE_NUM_COL = "virus_number"

CHANNELS_AND_PANNELS_INFO_PATH = f"/home/projects/hornsteinlab/Collaboration/Guy_Lior/fuNOVA_Screen_B{BATCH}/integrated_info.csv"  


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
        # get the column shift for the current part
        col_shift = plate_part_to_col_shift_dict[part]

        # extract the genes for the current part and plate
        genes = df_plate[df_plate['part'] == part][GENE_NAME_COL].tolist()

        # genes are in their real order, so we can enumerate them and assign the correct row index
        for row_idx, gene in enumerate(genes):
            # 2 reps per gene
            mapping[gene] = [(row_idx+1, col_idx_1 + col_shift), (row_idx+1, col_idx_2 + col_shift)]
    
    # all wells are from the same C9 cell line
    return {"C9": mapping}

def get_panel_markers(plate, panel):
    # Load the CSV file containing channel and panel information
    df = pd.read_csv(CHANNELS_AND_PANNELS_INFO_PATH)

    # Filter the DataFrame to get the row corresponding to the specified plate and panel
    row = df[(df['panel'] == f"Panel{panel}") & (df['plate'] == f"plate{plate}")]

    markers = []
    for ch in ["ch1", "ch2", "ch3", "ch4"]:
        # Get the marker for the current channel
        marker = row.get(f'{ch}_marker', None)
        if marker is None:
            raise ValueError(f"Marker for {ch} not found for plate {plate} and panel {panel}")
        markers.append(marker.values[0])

    # return a list with the markers in the correct order of ch1, ch2, ch3, ch4
    return markers


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
        self.col_rep1 = 3
        self.col_rep2 = 4
        self.panel = 'A'
        self.panel_number = panel_letter_to_num_dict[self.panel]
        self.markers = get_panel_markers(plate, self.panel_number)
        
        super().__init__(batch, plate)

class Config_B(Config_Base_4Markers):
    def __init__(self, batch, plate):

        # Params:
        self.col_rep1 = 1
        self.col_rep2 = 2
        self.panel = 'B'
        self.panel_number = panel_letter_to_num_dict[self.panel]
        self.markers = get_panel_markers(plate, self.panel_number)

        super().__init__(batch, plate)

class Config_C(Config_Base_4Markers):
    def __init__(self, batch, plate):

        # Params:
        self.col_rep1 = 7
        self.col_rep2 = 8
        self.panel = 'C'
        self.panel_number = panel_letter_to_num_dict[self.panel]
        # same marker for 4A and 4B
        self.markers = get_panel_markers(plate, f"{self.panel_number}A")

        super().__init__(batch, plate)
        self.INCLUDE_SUB_FOLDERS = [f'plate{self.PLATE}/Panel{self.panel_number}A', f'plate{self.PLATE}/Panel{self.panel_number}B']

class Config_D(Config_Base_4Markers):
    def __init__(self, batch, plate):

         # Params:
        self.col_rep1 = 5
        self.col_rep2 = 6
        self.panel = 'D'
        self.panel_number = panel_letter_to_num_dict[self.panel]
        self.markers = get_panel_markers(plate, self.panel_number)

        super().__init__(batch, plate)


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    df.to_csv(f"/home/projects/hornsteinlab/Collaboration/Guy_Lior/fuNOVA_Screen_B{BATCH}/virus_metadata_with_gene_names.csv", index=False)
    


