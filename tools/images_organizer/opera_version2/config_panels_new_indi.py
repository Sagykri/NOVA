######################################################
########## Please Don't Change This Section ##########
######################################################

import os
import sys

sys.path.insert(1, os.getenv("NOVA_HOME"))
from tools.images_organizer.opera_version2.config import Config

batch = 'batch8'

# 4 markers:
# [DAPI, Cy3, Cy2, Cy5]
# ch1 - DAPI
# ch2 - Cy3 (mCherry)
# ch3 - Cy2 (GFP)
# ch4 - Cy5

# 3 markers:
# [DAPI, Cy5, Cy3]
# ch1 - DAPI
# ch2 - Cy5 
# ch3 - Cy3 
# Expect for panelI where the order is [DAPI, Cy3, Cy5]

def get_mappings(row_rep1:int, row_rep2:int):
    return {
        "WT": {
            "stress":    [(row_rep1,1), (row_rep2,1)], 
            "Untreated": [(row_rep1,2), (row_rep2,2)]
        },
        "TDP43": {
            "Untreated": [(row_rep1,3), (row_rep2,3)]
        },
        "OPTN": {
            "Untreated": [(row_rep1,4), (row_rep2,4)]
        },
        "TBK1": {
            "Untreated": [(row_rep1,5), (row_rep2,5)]
        },
        "FUSHomozygous": { 
            "Untreated": [(row_rep1,6), (row_rep2,6)]
        },
        "FUSHeterozygous": {
            "Untreated": [(row_rep1,7), (row_rep2,7)]
        },
        "FUSRevertant": {
            "Untreated": [(row_rep1,8), (row_rep2,8)]
        },
        "SNCA": {
            "Untreated": [(row_rep1,9), (row_rep2,9)]
        }
    }

class Config_Base_4Markers(Config):
    def __init__(self):
        super().__init__()
        
        self.FOLDERS = [batch]
        self.INCLUDE_SUB_FOLDERS = [f'{batch}/Panel{self.panel}']
        self.CONFIG = {
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3", "ch4"],
            self.KEY_REPS: ["rep1", "rep2"],
            self.KEY_CELL_LINES: get_mappings(self.row_rep1, self.row_rep2),
            self.KEY_MARKERS: {f'panel{self.panel}': self.markers}
        }

class Config_Base_3Markers(Config_Base_4Markers):
    def __init__(self):
        super().__init__()

        self.CONFIG[self.KEY_MARKERS_ALIAS_ORDERED] = ["ch1", "ch2", "ch3"]

class Config_A(Config_Base_4Markers):
    def __init__(self):

        # Params:
        self.row_rep1 = 1
        self.row_rep2 = 2
        self.panel = 'A'
        # [DAPI, Cy3, Cy2, Cy5]
        self.markers = ["DAPI", "FMRP", "PURA", "G3BP1"]
        
        super().__init__()

#######################################

class Config_B(Config_Base_4Markers):
    def __init__(self):

        # Params:
        self.row_rep1 = 3
        self.row_rep2 = 4
        self.panel = 'B'
        # [DAPI, Cy3, Cy2, Cy5]
        self.markers = ["DAPI", "SON", "CD41", "NONO"]

        super().__init__()
        
#######################################

class Config_C(Config_Base_4Markers):
    def __init__(self):

        # Params:
        self.row_rep1 = 5
        self.row_rep2 = 6
        self.panel = 'C'
        # [DAPI, Cy3, Cy2, Cy5]
        self.markers = ["DAPI", "KIF5A", "Tubulin", "SQSTM1"]

        super().__init__()
        
#######################################

class Config_D(Config_Base_4Markers):
    def __init__(self):

        # Params:
        self.row_rep1 = 7
        self.row_rep2 = 8
        self.panel = 'D'
        # [DAPI, Cy3, Cy2, Cy5]
        self.markers = ["DAPI", "CLTC", "Phalloidin", "PSD95"]

        super().__init__()
        
#######################################

class Config_E(Config_Base_3Markers):
    def __init__(self):

        # Params:
        self.row_rep1 = 1
        self.row_rep2 = 2
        self.panel = 'E'
        # [DAPI, Cy5, Cy3]
        self.markers = ["DAPI", "NEMO", "DCP1A"]

        super().__init__()
        
#######################################

class Config_F(Config_Base_3Markers):
    def __init__(self):

        # Params:
        self.row_rep1 = 3
        self.row_rep2 = 4
        self.panel = 'F'
        # [DAPI, Cy5, Cy3]
        self.markers = ["DAPI", "GM130", "Calreticulin"]

        super().__init__()
        
#######################################

class Config_G(Config_Base_3Markers):
    def __init__(self):

        # Params:
        self.row_rep1 = 5
        self.row_rep2 = 6
        self.panel = 'G'
        # [DAPI, Cy5, Cy3]
        self.markers = ["DAPI", "NCL", "FUS"]

        super().__init__()
        
#######################################

class Config_H(Config_Base_3Markers):
    def __init__(self):

        # Params:
        self.row_rep1 = 7
        self.row_rep2 = 8
        self.panel = 'H'
        # [DAPI, Cy5, Cy3]
        self.markers = ["DAPI", "LSM14A", "HNRNPA1"]

        super().__init__()
        
#######################################

class Config_I(Config_Base_3Markers):
    def __init__(self):

        # Params:
        self.row_rep1 = 1
        self.row_rep2 = 2
        self.panel = 'I'
        # [DAPI, Cy3, Cy5] - CHANGED!!
        self.markers = ["DAPI", "PML",  "TDP43"]

        super().__init__()
        
#######################################

class Config_J(Config_Base_3Markers):
    def __init__(self):

        # Params:
        self.row_rep1 = 3
        self.row_rep2 = 4
        self.panel = 'J'
        # [DAPI, Cy5, Cy3]
        self.markers = ["DAPI", "ANXA11", "LAMP1"]

        super().__init__()
        
#######################################

class Config_K(Config_Base_3Markers):
    def __init__(self):

        # Params:
        self.row_rep1 = 5
        self.row_rep2 = 6
        self.panel = 'K'
        # [DAPI, Cy5, Cy3] - rechecked 30.6.25
        self.markers = ["DAPI", "PEX14", "SNCA"]

        super().__init__()
        
#######################################

class Config_L(Config_Base_4Markers):
    def __init__(self):

        # Params:
        self.row_rep1 = 7
        self.row_rep2 = 8
        self.panel = 'L'
        # [DAPI, Cy3, Cy2, Cy5]
        self.markers = ["DAPI", "TIA1", "TOMM20", "mitotracker"]

        super().__init__()
        
#######################################