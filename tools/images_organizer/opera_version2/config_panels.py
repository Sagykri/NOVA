######################################################
########## Please Don't Change This Section ##########
######################################################

import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from tools.images_organizer.opera_version2.config import Config

batch = 'batch2'

class Config_A(Config):
    def __init__(self):
        super().__init__()

        # folder_name = "PanelABC_Batch1"
        folder_name = batch
        self.FOLDERS = [folder_name]
        # self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/images']
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/PanelA']



        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        row_rep1 = 2
        row_rep2 = 3
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "WT": {
                    "stress": [(row_rep1,2), (row_rep2,2)], 
                    "Untreated": [(row_rep1,4), (row_rep2,4)]
                },
                "FUSHomozygous": {
                    "stress": [(row_rep1,3), (row_rep2,3)], 
                    "Untreated": [(row_rep1,5), (row_rep2,5)]
                },
                "TBK1": {
                    "Untreated": [(row_rep1,6), (row_rep2,6)]
                },
                "TDP43": {
                    "Untreated": [(row_rep1,7), (row_rep2,7)]
                },
                "FUSHeterozygous": {
                    "Untreated": [(row_rep1,8), (row_rep2,8)]
                },
                "FUSRevertant": {
                    "Untreated": [(row_rep1,9), (row_rep2,9)]
                },
                "OPTN": {
                    "Untreated": [(row_rep1,10), (row_rep2,10)]
                },
                "SNCA": {
                    "Untreated": [(row_rep1,11), (row_rep2,11)]
                }
            },
            # [DAPI, mCherry, GFP, Cy5]
            # ch1 - DAPI
            # ch2 - Cy3 (mCherry)
            # ch3 - Cy2 (GFP)
            # ch4 - Cy5
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3", "ch4"],
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
        row_rep1 = 4
        row_rep2 = 5
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "WT": {
                    "stress": [(row_rep1,2), (row_rep2,2)], 
                    "Untreated": [(row_rep1,4), (row_rep2,4)]
                },
                "FUSHomozygous": {
                    "stress": [(row_rep1,3), (row_rep2,3)], 
                    "Untreated": [(row_rep1,5), (row_rep2,5)]
                },
                "TBK1": {
                    "Untreated": [(row_rep1,6), (row_rep2,6)]
                },
                "TDP43": {
                    "Untreated": [(row_rep1,7), (row_rep2,7)]
                },
                "FUSHeterozygous": {
                    "Untreated": [(row_rep1,8), (row_rep2,8)]
                },
                "FUSRevertant": {
                    "Untreated": [(row_rep1,9), (row_rep2,9)]
                },
                "OPTN": {
                    "Untreated": [(row_rep1,10), (row_rep2,10)]
                },
                "SNCA": {
                    "Untreated": [(row_rep1,11), (row_rep2,11)]
                }
            },
            # [DAPI, mCherry, GFP, Cy5]
            # ch1 - DAPI
            # ch2 - Cy3 (mCherry)
            # ch3 - Cy2 (GFP)
            # ch4 - Cy5
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3", "ch4"],
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
        row_rep1 = 6
        row_rep2 = 7
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "WT": {
                    "stress": [(row_rep1,2), (row_rep2,2)], 
                    "Untreated": [(row_rep1,4), (row_rep2,4)]
                },
                "FUSHomozygous": {
                    "stress": [(row_rep1,3), (row_rep2,3)], 
                    "Untreated": [(row_rep1,5), (row_rep2,5)]
                },
                "TBK1": {
                    "Untreated": [(row_rep1,6), (row_rep2,6)]
                },
                "TDP43": {
                    "Untreated": [(row_rep1,7), (row_rep2,7)]
                },
                "FUSHeterozygous": {
                    "Untreated": [(row_rep1,8), (row_rep2,8)]
                },
                "FUSRevertant": {
                    "Untreated": [(row_rep1,9), (row_rep2,9)]
                },
                "OPTN": {
                    "Untreated": [(row_rep1,10), (row_rep2,10)]
                },
                "SNCA": {
                    "Untreated": [(row_rep1,11), (row_rep2,11)]
                }
            },
            # [DAPI, mCherry, GFP, Cy5]
            # ch1 - DAPI
            # ch2 - Cy3 (mCherry)
            # ch3 - Cy2 (GFP)
            # ch4 - Cy5
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3", "ch4"],
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
        row_rep1 = 2
        row_rep2 = 3
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "WT": {
                    "stress": [(row_rep1,2), (row_rep2,2)], 
                    "Untreated": [(row_rep1,4), (row_rep2,4)]
                },
                "FUSHomozygous": {
                    "stress": [(row_rep1,3), (row_rep2,3)], 
                    "Untreated": [(row_rep1,5), (row_rep2,5)]
                },
                "TBK1": {
                    "Untreated": [(row_rep1,6), (row_rep2,6)]
                },
                "TDP43": {
                    "Untreated": [(row_rep1,7), (row_rep2,7)]
                },
                "FUSHeterozygous": {
                    "Untreated": [(row_rep1,8), (row_rep2,8)]
                },
                "FUSRevertant": {
                    "Untreated": [(row_rep1,9), (row_rep2,9)]
                },
                "OPTN": {
                    "Untreated": [(row_rep1,10), (row_rep2,10)]
                },
                "SNCA": {
                    "Untreated": [(row_rep1,11), (row_rep2,11)]
                }
            },
            # [DAPI, Cy5, mCherry]
            # ch1 - DAPI
            # ch2 - Cy5
            # ch3 - Cy3 (mCherry)
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3"],
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
        row_rep1 = 4
        row_rep2 = 5
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "WT": {
                    "stress": [(row_rep1,2), (row_rep2,2)], 
                    "Untreated": [(row_rep1,4), (row_rep2,4)]
                },
                "FUSHomozygous": {
                    "stress": [(row_rep1,3), (row_rep2,3)], 
                    "Untreated": [(row_rep1,5), (row_rep2,5)]
                },
                "TBK1": {
                    "Untreated": [(row_rep1,6), (row_rep2,6)]
                },
                "TDP43": {
                    "Untreated": [(row_rep1,7), (row_rep2,7)]
                },
                "FUSHeterozygous": {
                    "Untreated": [(row_rep1,8), (row_rep2,8)]
                },
                "FUSRevertant": {
                    "Untreated": [(row_rep1,9), (row_rep2,9)]
                },
                "OPTN": {
                    "Untreated": [(row_rep1,10), (row_rep2,10)]
                },
                "SNCA": {
                    "Untreated": [(row_rep1,11), (row_rep2,11)]
                }
            },
            # [DAPI, Cy5, mCherry]
            # ch1 - DAPI
            # ch2 - Cy5
            # ch3 - Cy3 (mCherry)
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3"],
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
        row_rep1 = 6
        row_rep2 = 7
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "WT": {
                    "stress": [(row_rep1,2), (row_rep2,2)], 
                    "Untreated": [(row_rep1,4), (row_rep2,4)]
                },
                "FUSHomozygous": {
                    "stress": [(row_rep1,3), (row_rep2,3)], 
                    "Untreated": [(row_rep1,5), (row_rep2,5)]
                },
                "TBK1": {
                    "Untreated": [(row_rep1,6), (row_rep2,6)]
                },
                "TDP43": {
                    "Untreated": [(row_rep1,7), (row_rep2,7)]
                },
                "FUSHeterozygous": {
                    "Untreated": [(row_rep1,8), (row_rep2,8)]
                },
                "FUSRevertant": {
                    "Untreated": [(row_rep1,9), (row_rep2,9)]
                },
                "OPTN": {
                    "Untreated": [(row_rep1,10), (row_rep2,10)]
                },
                "SNCA": {
                    "Untreated": [(row_rep1,11), (row_rep2,11)]
                }
            },
            # [DAPI, Cy5, mCherry]
            # ch1 - DAPI
            # ch2 - Cy5
            # ch3 - Cy3 (mCherry)
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3"],
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
        row_rep1 = 2
        row_rep2 = 3
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "WT": {
                    "stress": [(row_rep1,2), (row_rep2,2)], 
                    "Untreated": [(row_rep1,4), (row_rep2,4)]
                },
                "FUSHomozygous": {
                    "stress": [(row_rep1,3), (row_rep2,3)], 
                    "Untreated": [(row_rep1,5), (row_rep2,5)]
                },
                "TBK1": {
                    "Untreated": [(row_rep1,6), (row_rep2,6)]
                },
                "TDP43": {
                    "Untreated": [(row_rep1,7), (row_rep2,7)]
                },
                "FUSHeterozygous": {
                    "Untreated": [(row_rep1,8), (row_rep2,8)]
                },
                "FUSRevertant": {
                    "Untreated": [(row_rep1,9), (row_rep2,9)]
                },
                "OPTN": {
                    "Untreated": [(row_rep1,10), (row_rep2,10)]
                },
                "SNCA": {
                    "Untreated": [(row_rep1,11), (row_rep2,11)]
                }
            },
            # [DAPI, Cy5, mCherry]
            # ch1 - DAPI
            # ch2 - Cy5
            # ch3 - Cy3 (mCherry)
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3"],
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
        row_rep1 = 4
        row_rep2 = 5
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "WT": {
                    "stress": [(row_rep1,2), (row_rep2,2)], 
                    "Untreated": [(row_rep1,4), (row_rep2,4)]
                },
                "FUSHomozygous": {
                    "stress": [(row_rep1,3), (row_rep2,3)], 
                    "Untreated": [(row_rep1,5), (row_rep2,5)]
                },
                "TBK1": {
                    "Untreated": [(row_rep1,6), (row_rep2,6)]
                },
                "TDP43": {
                    "Untreated": [(row_rep1,7), (row_rep2,7)]
                },
                "FUSHeterozygous": {
                    "Untreated": [(row_rep1,8), (row_rep2,8)]
                },
                "FUSRevertant": {
                    "Untreated": [(row_rep1,9), (row_rep2,9)]
                },
                "OPTN": {
                    "Untreated": [(row_rep1,10), (row_rep2,10)]
                },
                "SNCA": {
                    "Untreated": [(row_rep1,11), (row_rep2,11)]
                }
            },
            # [DAPI, Cy5, mCherry]
            # ch1 - DAPI
            # ch2 - Cy5
            # ch3 - Cy3 (mCherry)
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3"],
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
        row_rep1 = 6
        row_rep2 = 7
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "WT": {
                    "stress": [(row_rep1,2), (row_rep2,2)], 
                    "Untreated": [(row_rep1,4), (row_rep2,4)]
                },
                "FUSHomozygous": {
                    "stress": [(row_rep1,3), (row_rep2,3)], 
                    "Untreated": [(row_rep1,5), (row_rep2,5)]
                },
                "TBK1": {
                    "Untreated": [(row_rep1,6), (row_rep2,6)]
                },
                "TDP43": {
                    "Untreated": [(row_rep1,7), (row_rep2,7)]
                },
                "FUSHeterozygous": {
                    "Untreated": [(row_rep1,8), (row_rep2,8)]
                },
                "FUSRevertant": {
                    "Untreated": [(row_rep1,9), (row_rep2,9)]
                },
                "OPTN": {
                    "Untreated": [(row_rep1,10), (row_rep2,10)]
                },
                "SNCA": {
                    "Untreated": [(row_rep1,11), (row_rep2,11)]
                }
            },
            # [DAPI, Cy5, mCherry]
            # ch1 - DAPI
            # ch2 - Cy5
            # ch3 - Cy3 (mCherry)
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3"],
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
        row_rep1 = 2
        row_rep2 = 3
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "WT": {
                    "stress": [(row_rep1,2), (row_rep2,2)], 
                    "Untreated": [(row_rep1,4), (row_rep2,4)]
                },
                "FUSHomozygous": {
                    "stress": [(row_rep1,3), (row_rep2,3)], 
                    "Untreated": [(row_rep1,5), (row_rep2,5)]
                },
                "TBK1": {
                    "Untreated": [(row_rep1,6), (row_rep2,6)]
                },
                "TDP43": {
                    "Untreated": [(row_rep1,7), (row_rep2,7)]
                },
                "FUSHeterozygous": {
                    "Untreated": [(row_rep1,8), (row_rep2,8)]
                },
                "FUSRevertant": {
                    "Untreated": [(row_rep1,9), (row_rep2,9)]
                },
                "OPTN": {
                    "Untreated": [(row_rep1,10), (row_rep2,10)]
                },
                "SNCA": {
                    "Untreated": [(row_rep1,11), (row_rep2,11)]
                }
            },
            # [DAPI, mCherry, GFP, Cy5]
            # ch1 - DAPI
            # ch2 - Cy3 (mCherry)
            # ch3 - Cy2 (GFP)
            # ch4 - Cy5
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3", "ch4"],
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
        row_rep1 = 4
        row_rep2 = 5
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "WT": {
                    "stress": [(row_rep1,2), (row_rep2,2)], 
                    "Untreated": [(row_rep1,4), (row_rep2,4)]
                },
                "FUSHomozygous": {
                    "stress": [(row_rep1,3), (row_rep2,3)], 
                    "Untreated": [(row_rep1,5), (row_rep2,5)]
                },
                "TBK1": {
                    "Untreated": [(row_rep1,6), (row_rep2,6)]
                },
                "TDP43": {
                    "Untreated": [(row_rep1,7), (row_rep2,7)]
                },
                "FUSHeterozygous": {
                    "Untreated": [(row_rep1,8), (row_rep2,8)]
                },
                "FUSRevertant": {
                    "Untreated": [(row_rep1,9), (row_rep2,9)]
                },
                "OPTN": {
                    "Untreated": [(row_rep1,10), (row_rep2,10)]
                },
                "SNCA": {
                    "Untreated": [(row_rep1,11), (row_rep2,11)]
                }
            },
            # [DAPI, mCherry, GFP, Cy5]
            # ch1 - DAPI
            # ch2 - Cy3 (mCherry)
            # ch3 - Cy2 (GFP) - NONE
            # ch4 - Cy5
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3", "ch4"],
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
        row_rep1 = 6
        row_rep2 = 7
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "WT": {
                    "stress": [(row_rep1,2), (row_rep2,2)], 
                    "Untreated": [(row_rep1,4), (row_rep2,4)]
                },
                "FUSHomozygous": {
                    "stress": [(row_rep1,3), (row_rep2,3)], 
                    "Untreated": [(row_rep1,5), (row_rep2,5)]
                },
                "TBK1": {
                    "Untreated": [(row_rep1,6), (row_rep2,6)]
                },
                "TDP43": {
                    "Untreated": [(row_rep1,7), (row_rep2,7)]
                },
                "FUSHeterozygous": {
                    "Untreated": [(row_rep1,8), (row_rep2,8)]
                },
                "FUSRevertant": {
                    "Untreated": [(row_rep1,9), (row_rep2,9)]
                },
                "OPTN": {
                    "Untreated": [(row_rep1,10), (row_rep2,10)]
                },
                "SNCA": {
                    "Untreated": [(row_rep1,11), (row_rep2,11)]
                }
            },
            # [DAPI, mCherry, GFP, Cy5]
            # ch1 - DAPI
            # ch2 - Cy3 (mCherry)
            # ch3 - Cy2 (GFP)
            # ch4 - Cy5
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3", "ch4"],
            self.KEY_MARKERS: {
                'panelL': ["DAPI", "PML", "PEX14", "mitotracker"]
                },
            self.KEY_REPS: ["rep1", "rep2"],
        }

#######################################