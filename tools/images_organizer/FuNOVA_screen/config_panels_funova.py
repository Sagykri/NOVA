######################################################
########## Please Don't Change This Section ##########
######################################################

import os
import sys
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))

from tools.images_organizer.FuNOVA_screen.config import Config
import pandas as pd

GENES_METADATA_PATH = "/home/projects/hornsteinlab/Collaboration/Guy_Lior/fuNOVA_Screen_B1/virus_metadata.csv"
GENE_NAME_COL = "virus_name"
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



def get_mappings(col_idx_1, col_idx_2, plate):
    # read the metadta file for given plate
    df = pd.read_csv(GENES_METADATA_PATH)
    df_plate = df[df['plate'] == plate]
    
    #remove nan values
    df_plate = df_plate.dropna(subset=[GENE_NAME_COL])
    
    unique_parts = df_plate['part'].unique()
    mapping = {}
    # for each part, get the genes and map them to their respective row and column indices
    for part in unique_parts:
            # each part has a different column shift (1-8, 9-16, 17-24) based on the plate division
            col_shift = plate_part_to_col_shift_dict[part]

            # get the genes for the current part and map them to their respective row and column indices
            genes = df_plate[df_plate['part'] == part][GENE_NAME_COL].tolist()
            for row_idx, gene in enumerate(genes):
                mapping[gene] = [(row_idx+1, col_idx_1 + col_shift), (row_idx+1, col_idx_2 + col_shift)]
    
    return {"C9": mapping}

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
        self.markers = ["FK-2", "p62","DAPI", "TDP-43"]
        
        super().__init__(batch, plate)

class Config_B(Config_Base_4Markers):
    def __init__(self, batch, plate):

        # Params:
        self.col_rep1 = 3
        self.col_rep2 = 4
        self.panel = 'B'
        self.markers = ["HDGFL2", "ATF6", "DAPI", "pTDP-43"]

        super().__init__(batch, plate)

class Config_C(Config_Base_4Markers):
    def __init__(self, batch, plate):

        # Params:
        self.col_rep1 = 5
        self.col_rep2 = 6
        self.panel = 'C'
        self.markers = ["G3BP1", "Calreticulin", "DAPI", "pAMPK"]

        super().__init__(batch, plate)

class Config_D(Config_Base_4Markers):
    def __init__(self, batch, plate):

        # Params:
        self.col_rep1 = 7
        self.col_rep2 = 8
        self.panel = 'D'
        self.markers = ["Aggreagtes", "pS6", "DAPI", "Cas3"]

        super().__init__(batch, plate)
        panel_number = panel_letter_to_num_dict[self.panel]
        self.INCLUDE_SUB_FOLDERS = [f'plate{self.PLATE}/Panel{panel_number}A', f'plate{self.PLATE}/Panel{panel_number}B']


if __name__ == "__main__":
    mapping = get_mappings(1, 2, 1)
    print(mapping)
        


