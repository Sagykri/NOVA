######################################################
########## Please Don't Change This Section ##########
######################################################

import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from tools.images_organizer.neurons_spd18days_version.config import Config

batch = "batch2"
cell_lines_config = {
                "WT": {
                    "stress": [(1,100), (101,200)], 
                    "Untreated": [(401,500), (501,600)]
                },
                "FUSHomozygous": {
                    "stress": [(301,400), (201,300)], 
                    "Untreated": [(701,800), (601,700)]
                },
                "TBK1": {
                    "Untreated": [(801,900), (901,1000)]
                },
                "TDP43": {
                    "Untreated": [(1101,1200), (1001,1100)]
                },
                "FUSHeterozygous": {
                    "Untreated": [(1201,1300), (1301,1400)]
                },
                "FUSRevertant": {
                    "Untreated": [(1501,1600), (1401,1500)]
                },
                "OPTN": {
                    "Untreated": [(1601,1700), (1701,1800)]
                },
                "SNCA": {
                    "Untreated": [(1901,2000), (1801,1900)]
                }
}
markers_alias_ordered = ["DAPI", "mCherry", "GFP", "Cy5"]
markers_alias_ordered2 = ["DAPI", "Cy5", "mCherry"]

#######################################

class Config_A(Config):
    def __init__(self):
        super().__init__()

        folder_name = batch
        self.FOLDERS = [folder_name]
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/PanelA']


        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
     
        
        self.CONFIG = {
            self.KEY_CELL_LINES: cell_lines_config,
            # [DAPI, mCherry, GFP, Cy5]
            # ch01 - DAPI
            # ch02 - Cy3 (mCherry)
            # ch03 - Cy2 (GFP)
            # ch04 - Cy5
            self.KEY_MARKERS_ALIAS_ORDERED: markers_alias_ordered,
            self.KEY_MARKERS: {
                'panelA': ["DAPI", "TOMM20", "PURA", "G3BP1"]
                },
            self.KEY_REPS: ["rep1", "rep2"],
        }

#######################################

class Config_B(Config):
    def __init__(self):
        super().__init__()

        folder_name = batch
        self.FOLDERS = [folder_name]
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/PanelB']



        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        self.CONFIG = {
            self.KEY_CELL_LINES: cell_lines_config,
            # [DAPI, mCherry, GFP, Cy5]
            # ch01 - DAPI
            # ch02 - Cy3 (mCherry)
            # ch03 - Cy2 (GFP)
            # ch04 - Cy5
            self.KEY_MARKERS_ALIAS_ORDERED: markers_alias_ordered,
            self.KEY_MARKERS: {
                'panelB': ["DAPI", "TDP43", "CD41", "AGO2"]
                },
            self.KEY_REPS: ["rep1", "rep2"],
        }

#######################################

class Config_C(Config):
    def __init__(self):
        super().__init__()

        folder_name = batch
        self.FOLDERS = [folder_name]
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/PanelC']



        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        self.CONFIG = {
            self.KEY_CELL_LINES: cell_lines_config,
            # [DAPI, mCherry, GFP, Cy5]
            # ch01 - DAPI
            # ch02 - Cy3 (mCherry)
            # ch03 - Cy2 (GFP)
            # ch04 - Cy5
            self.KEY_MARKERS_ALIAS_ORDERED: markers_alias_ordered,
            self.KEY_MARKERS: {
                'panelC': ["DAPI", "FMRP", "Phalloidin", "SQSTM1"]
                },
            self.KEY_REPS: ["rep1", "rep2"],
        }

#######################################


class Config_D(Config):
    def __init__(self):
        super().__init__()

        folder_name = batch
        self.FOLDERS = [folder_name]
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/PanelD']



        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        self.CONFIG = {
            self.KEY_CELL_LINES: cell_lines_config,
            # [DAPI, Cy5, mCherry]
            # ch01 - DAPI
            # ch02 - Cy5
            # ch03 - Cy3 (mCherry)
            self.KEY_MARKERS_ALIAS_ORDERED: markers_alias_ordered2,
            self.KEY_MARKERS: {
                'panelD': ["DAPI", "PSD95", "CLTC"]
                },
            self.KEY_REPS: ["rep1", "rep2"],
        }

#######################################


class Config_E(Config):
    def __init__(self):
        super().__init__()

        folder_name = batch
        self.FOLDERS = [folder_name]
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/PanelE']



        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        self.CONFIG = {
            self.KEY_CELL_LINES: cell_lines_config,
            # [DAPI, Cy5, mCherry]
            # ch01 - DAPI
            # ch02 - Cy5
            # ch03 - Cy3 (mCherry)
            self.KEY_MARKERS_ALIAS_ORDERED: markers_alias_ordered2,
            self.KEY_MARKERS: {
                'panelE': ["DAPI", "NEMO", "DCP1A"]
                },
            self.KEY_REPS: ["rep1", "rep2"],
        }

#######################################

class Config_F(Config):
    def __init__(self):
        super().__init__()

        folder_name = batch
        self.FOLDERS = [folder_name]
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/PanelF']



        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        self.CONFIG = {
            self.KEY_CELL_LINES: cell_lines_config,
            # [DAPI, Cy5, mCherry]
            # ch01 - DAPI
            # ch02 - Cy5
            # ch03 - Cy3 (mCherry)
            self.KEY_MARKERS_ALIAS_ORDERED: markers_alias_ordered2,
            self.KEY_MARKERS: {
                'panelF': ["DAPI", "GM130", "KIF5A"]
                },
            self.KEY_REPS: ["rep1", "rep2"],
        }

#######################################


class Config_G(Config):
    def __init__(self):
        super().__init__()

        folder_name = batch
        self.FOLDERS = [folder_name]
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/PanelG']



        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        self.CONFIG = {
            self.KEY_CELL_LINES: cell_lines_config,
            # [DAPI, Cy5, mCherry]
            # ch01 - DAPI
            # ch02 - Cy5
            # ch03 - Cy3 (mCherry)
            self.KEY_MARKERS_ALIAS_ORDERED: markers_alias_ordered2,
            self.KEY_MARKERS: {
                'panelG': ["DAPI", "FUS", "NCL"]
                },
            self.KEY_REPS: ["rep1", "rep2"],
        }

#######################################

class Config_H(Config):
    def __init__(self):
        super().__init__()

        folder_name = batch
        self.FOLDERS = [folder_name]
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/PanelH']



        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        self.CONFIG = {
            self.KEY_CELL_LINES: cell_lines_config,
            # [DAPI, Cy5, mCherry]
            # ch01 - DAPI
            # ch02 - Cy5
            # ch03 - Cy3 (mCherry)
            self.KEY_MARKERS_ALIAS_ORDERED: markers_alias_ordered2,
            self.KEY_MARKERS: {
                'panelH': ["DAPI", "SNCA", "ANXA11"]
                },
            self.KEY_REPS: ["rep1", "rep2"],
        }

#######################################

class Config_I(Config):
    def __init__(self):
        super().__init__()

        folder_name = batch
        self.FOLDERS = [folder_name]
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/PanelI']



        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        self.CONFIG = {
            self.KEY_CELL_LINES: cell_lines_config,
            # [DAPI, Cy5, mCherry]
            # ch01 - DAPI
            # ch02 - Cy5
            # ch03 - Cy3 (mCherry)
            self.KEY_MARKERS_ALIAS_ORDERED: markers_alias_ordered2,
            self.KEY_MARKERS: {
                'panelI': ["DAPI", "LAMP1", "Calreticulin"]
                },
            self.KEY_REPS: ["rep1", "rep2"],
        }

#######################################

class Config_J(Config):
    def __init__(self):
        super().__init__()

        folder_name = batch
        self.FOLDERS = [folder_name]
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/PanelJ']



        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        self.CONFIG = {
            self.KEY_CELL_LINES: cell_lines_config,
            # [DAPI, mCherry, GFP, Cy5]
            # ch01 - DAPI
            # ch02 - Cy3 (mCherry)
            # ch03 - Cy2 (GFP)
            # ch04 - Cy5
            self.KEY_MARKERS_ALIAS_ORDERED: markers_alias_ordered,
            self.KEY_MARKERS: {
                'panelJ': ["DAPI", "PSPC1", "Tubulin", "HNRNPA1"]
                },
            self.KEY_REPS: ["rep1", "rep2"],
        }

#######################################

class Config_K(Config):
    def __init__(self):
        super().__init__()

        folder_name = batch
        self.FOLDERS = [folder_name]
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/PanelK']



        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        self.CONFIG = {
            self.KEY_CELL_LINES: cell_lines_config,
            # [DAPI, mCherry, GFP, Cy5]
            # ch01 - DAPI
            # ch02 - Cy3 (mCherry)
            # ch03 - Cy2 (GFP) - NONE
            # ch04 - Cy5
            self.KEY_MARKERS_ALIAS_ORDERED: markers_alias_ordered,
            self.KEY_MARKERS: {
                'panelK': ["DAPI", "VDAC1", None, "NONO"]
                },
            self.KEY_REPS: ["rep1", "rep2"],
        }

#######################################

class Config_L(Config):
    def __init__(self):
        super().__init__()

        folder_name = batch
        self.FOLDERS = [folder_name]
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/PanelL']



        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        self.CONFIG = {
            self.KEY_CELL_LINES: cell_lines_config,
            # [DAPI, mCherry, GFP, Cy5]
            # ch01 - DAPI
            # ch02 - Cy3 (mCherry)
            # ch03 - Cy2 (GFP)
            # ch04 - Cy5
            self.KEY_MARKERS_ALIAS_ORDERED: markers_alias_ordered,
            self.KEY_MARKERS: {
                'panelL': ["DAPI", "PML", "PEX14", "mitotracker"]
                },
            self.KEY_REPS: ["rep1", "rep2"],
        }

#######################################