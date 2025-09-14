######################################################
########## Please Don't Change This Section ##########
######################################################

import os
import sys

sys.path.insert(1, os.getenv("NOVA_HOME"))
from tools.images_organizer.FuNOVA_Guy_Lior.config import Config

ROW_MAP = {
    "PPP2R1A": 1,
    "HMGCS1": 3,
    "PIK3C3": 5,
    "NDUFAB1": 7,
    "MAPKAP1": 9,
    "NDUFS2": 11,
    "RALA": 13,
    "TLK1": 15,
    "NRIP1": 1,
    "TARDBP": 3,
    "RANBP17": 5,
    "CYLD": 7,
    "NT-1873": 9,
    "NT-6301-3085": 11,
    "Intergenic": 13,
    "Untreated": 15
}

def get_mappings(col_idx_1, col_idx_2, col_shift):
    return {
        "CTL": {
            "PPP2R1A":          [(ROW_MAP["PPP2R1A"],col_idx_1), 
                                (ROW_MAP["PPP2R1A"],col_idx_2)],
            "HMGCS1":           [(ROW_MAP["HMGCS1"],col_idx_1), 
                                (ROW_MAP["HMGCS1"],col_idx_2)],
            "PIK3C3":           [(ROW_MAP["PIK3C3"],col_idx_1), 
                                (ROW_MAP["PIK3C3"],col_idx_2)],
            "NDUFAB1":          [(ROW_MAP["NDUFAB1"],col_idx_1), 
                                (ROW_MAP["NDUFAB1"],col_idx_2)],
            "MAPKAP1":          [(ROW_MAP["MAPKAP1"],col_idx_1), 
                                (ROW_MAP["MAPKAP1"],col_idx_2)],
            "NDUFS2":           [(ROW_MAP["NDUFS2"],col_idx_1), 
                                (ROW_MAP["NDUFS2"],col_idx_2)],
            "RALA":             [(ROW_MAP["RALA"],col_idx_1), 
                                (ROW_MAP["RALA"],col_idx_2)],
            "TLK1":             [(ROW_MAP["TLK1"],col_idx_1), 
                                (ROW_MAP["TLK1"],col_idx_2)],
            "NRIP1":            [(ROW_MAP["NRIP1"],col_idx_1 + col_shift), 
                                (ROW_MAP["NRIP1"],col_idx_2 + col_shift)],
            "TARDBP":           [(ROW_MAP["TARDBP"],col_idx_1 + col_shift), 
                                (ROW_MAP["TARDBP"],col_idx_2 + col_shift)],
            "RANBP17":          [(ROW_MAP["RANBP17"],col_idx_1 + col_shift), 
                                (ROW_MAP["RANBP17"],col_idx_2 + col_shift)],
            "CYLD":             [(ROW_MAP["CYLD"],col_idx_1 + col_shift), 
                                (ROW_MAP["CYLD"],col_idx_2 + col_shift)],
            "NT-1873":          [(ROW_MAP["NT-1873"],col_idx_1 + col_shift), 
                                (ROW_MAP["NT-1873"],col_idx_2 + col_shift)],
            "NT-6301-3085":    [(ROW_MAP["NT-6301-3085"],col_idx_1 + col_shift), 
                                 (ROW_MAP["NT-6301-3085"],col_idx_2 + col_shift)],
            "Intergenic":       [(ROW_MAP["Intergenic"],col_idx_1 + col_shift), 
                                (ROW_MAP["Intergenic"],col_idx_2 + col_shift)],
            "Untreated":        [(ROW_MAP["Untreated"],col_idx_1 + col_shift), 
                                (ROW_MAP["Untreated"],col_idx_2 + col_shift)]
        },
        "C9": {
            "PPP2R1A":          [(ROW_MAP["PPP2R1A"] + 1 ,col_idx_1), 
                                (ROW_MAP["PPP2R1A"] + 1,col_idx_2)],
            "HMGCS1":           [(ROW_MAP["HMGCS1"] + 1,col_idx_1), 
                                (ROW_MAP["HMGCS1"] + 1,col_idx_2)],
            "PIK3C3":           [(ROW_MAP["PIK3C3"] + 1,col_idx_1), 
                                (ROW_MAP["PIK3C3"] + 1,col_idx_2)],
            "NDUFAB1":          [(ROW_MAP["NDUFAB1"] + 1,col_idx_1), 
                                (ROW_MAP["NDUFAB1"] + 1,col_idx_2)],
            "MAPKAP1":          [(ROW_MAP["MAPKAP1"] + 1,col_idx_1), 
                                (ROW_MAP["MAPKAP1"] + 1,col_idx_2)],
            "NDUFS2":           [(ROW_MAP["NDUFS2"] + 1,col_idx_1), 
                                (ROW_MAP["NDUFS2"] + 1,col_idx_2)],
            "RALA":             [(ROW_MAP["RALA"] + 1,col_idx_1), 
                                (ROW_MAP["RALA"] + 1,col_idx_2)],
            "TLK1":             [(ROW_MAP["TLK1"] + 1,col_idx_1), 
                                (ROW_MAP["TLK1"] + 1,col_idx_2)],
            "NRIP1":            [(ROW_MAP["NRIP1"] + 1,col_idx_1 + col_shift), 
                                (ROW_MAP["NRIP1"] + 1,col_idx_2 + col_shift)],
            "TARDBP":           [(ROW_MAP["TARDBP"] + 1,col_idx_1 + col_shift), 
                                (ROW_MAP["TARDBP"] + 1,col_idx_2 + col_shift)],
            "RANBP17":          [(ROW_MAP["RANBP17"] + 1,col_idx_1 + col_shift), 
                                (ROW_MAP["RANBP17"] + 1,col_idx_2 + col_shift)],
            "CYLD":             [(ROW_MAP["CYLD"] + 1,col_idx_1 + col_shift), 
                                (ROW_MAP["CYLD"] + 1,col_idx_2 + col_shift)],
            "NT-1873":          [(ROW_MAP["NT-1873"] + 1,col_idx_1 + col_shift), 
                                (ROW_MAP["NT-1873"] + 1,col_idx_2 + col_shift)],
            "NT-6301-3085":     [(ROW_MAP["NT-6301-3085"] + 1,col_idx_1 + col_shift), 
                                 (ROW_MAP["NT-6301-3085"] + 1,col_idx_2 + col_shift)],
            "Intergenic":       [(ROW_MAP["Intergenic"] + 1,col_idx_1 + col_shift), 
                                (ROW_MAP["Intergenic"] + 1,col_idx_2 + col_shift)],
            "Untreated":        [(ROW_MAP["Untreated"] + 1,col_idx_1 + col_shift), 
                                (ROW_MAP["Untreated"] + 1,col_idx_2 + col_shift)]
        },
    }

class Config_Base_Data(Config):
    def __init__(self, batch):
        super().__init__()

        self.BATCH = batch
        self.TOTAL_NUM_COLUMNS = 24
        self.HALF_PLATE_NUM_COLUMNS = self.TOTAL_NUM_COLUMNS // 2

class Config_Base_4Markers(Config_Base_Data):
    def __init__(self, batch):
        super().__init__(batch)

        self.FOLDERS = [f"Plate_{self.BATCH}"]
        self.INCLUDE_SUB_FOLDERS = [f'Plate_{self.BATCH}/Panel{self.panel}']
        self.CONFIG = {
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3", "ch4"],
            self.KEY_REPS: ["rep1", "rep2"],
            self.KEY_CELL_LINES: get_mappings(self.col_rep1, self.col_rep2, self.HALF_PLATE_NUM_COLUMNS),
            self.KEY_MARKERS: {f'panel{self.panel}': self.markers}
        }


class Config_A(Config_Base_4Markers):
    def __init__(self, batch):
        
        # Params:
        self.col_rep1 = 1
        self.col_rep2 = 2
        self.panel = 'A'
        self.markers = ["DAPI", "Cas3", "FK-2", "SMI32"]
        
        super().__init__(batch)

class Config_B(Config_Base_4Markers):
    def __init__(self, batch):

        # Params:
        self.col_rep1 = 3
        self.col_rep2 = 4
        self.panel = 'B'
        self.markers = ["DAPI", "pDRP1", "TOMM20", "pCaMKIIa"]

        super().__init__(batch)

class Config_C(Config_Base_4Markers):
    def __init__(self, batch):

        # Params:
        self.col_rep1 = 5
        self.col_rep2 = 6
        self.panel = 'C'
        self.markers = ["DAPI", "pTDP-43", "TDP-43", "ATF6"]

        super().__init__(batch)

class Config_D(Config_Base_4Markers):
    def __init__(self, batch):

        # Params:
        self.col_rep1 = 7
        self.col_rep2 = 8
        self.panel = 'D'
        self.markers = ["DAPI", "pAMPK", "HDGFL2", "pS6"]

        super().__init__(batch)

class Config_E(Config_Base_4Markers):
    def __init__(self, batch):

        # Params:
        self.col_rep1 = 9
        self.col_rep2 = 10
        self.panel = 'E'
        self.markers = ["DAPI", "PAR", "UNC13A", "Calreticulin"]

        super().__init__(batch)

class Config_F(Config_Base_4Markers):
    def __init__(self, batch):

        # Params:
        self.col_rep1 =11
        self.col_rep2 = 12
        self.panel = 'F'
        self.markers = ["DAPI", "LC3-II", "p62", "CathepsinD"]

        super().__init__(batch)

        self.INCLUDE_SUB_FOLDERS = [f'Plate_{self.BATCH}/Panel{self.panel}1', f'Plate_{self.BATCH}/Panel{self.panel}2']
        
        
