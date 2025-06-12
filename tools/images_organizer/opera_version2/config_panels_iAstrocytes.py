######################################################
########## Please Don't Change This Section ##########
######################################################

import os
import sys

sys.path.insert(1, os.getenv("NOVA_HOME"))
from tools.images_organizer.opera_version2.config import Config

row_rep1 = 2
row_rep2 = 3
row_rep3 = 4

# [488, 568, DAPI, 647]
# ch1 - 488
# ch2 - 568
# ch3 - DAPI
# ch4 - 647

class Config_WT(Config):
    def __init__(self):
        super().__init__()

        self.folder_name = "batch1/161"
        self.FOLDERS = [self.folder_name]
        

class Config_C9(Config):
    def __init__(self):
        super().__init__()

        self.folder_name = "batch1/78"
        self.FOLDERS = [self.folder_name]
        
#######################################
        
#######################################

class Config_WT_PanelA(Config_WT):
    def __init__(self):
        super().__init__()

        self.INCLUDE_SUB_FOLDERS = [f'{self.folder_name}/PanelA']
        col_number = 2

        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "WT": {
                    "Untreated": [(row_rep1, col_number), (row_rep2, col_number), (row_rep3, col_number)]
                }
            },
            
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3", "ch4"],
            self.KEY_MARKERS: {
                'panelA': ["Vimentin", "WDR49", "DAPI", "PML"]
                },
            self.KEY_REPS: ["rep1", "rep2", "rep3"],
        }

#######################################


class Config_WT_PanelB(Config_WT):
    def __init__(self):
        super().__init__()

        self.INCLUDE_SUB_FOLDERS = [f'{self.folder_name}/PanelB']
        col_number = 3

        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "WT": {
                    "Untreated": [(row_rep1, col_number), (row_rep2, col_number), (row_rep3, col_number)]
                }
            },
            
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3", "ch4"],
            self.KEY_MARKERS: {
                'panelB': ["ARL13B", "WDR49", "DAPI", "Vimentin"]
                },
            self.KEY_REPS: ["rep1", "rep2", "rep3"],
        }

#######################################

class Config_WT_PanelC(Config_WT):
    def __init__(self):
        super().__init__()

        self.INCLUDE_SUB_FOLDERS = [f'{self.folder_name}/PanelC']
        col_number = 6

        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "WT": {
                    "Untreated": [(row_rep1, col_number), (row_rep2, col_number), (row_rep3, col_number)]
                }
            },
            
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3", "ch4"],
            self.KEY_MARKERS: {
                'panelC': ["Vimentin", "Calreticulin", "DAPI", "PML"]
                },
            self.KEY_REPS: ["rep1", "rep2", "rep3"],
        }

#######################################

class Config_WT_PanelD(Config_WT):
    def __init__(self):
        super().__init__()

        self.INCLUDE_SUB_FOLDERS = [f'{self.folder_name}/PanelD']
        col_number = 7

        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "WT": {
                    "Untreated": [(row_rep1, col_number), (row_rep2, col_number), (row_rep3, col_number)]
                }
            },
            
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3", "ch4"],
            self.KEY_MARKERS: {
                'panelD': ["Vimentin", "WDR49", "DAPI", "TDP43"]
                },
            self.KEY_REPS: ["rep1", "rep2", "rep3"],
        }

################################################################################################

class Config_C9_PanelA(Config_C9):
    def __init__(self):
        super().__init__()

        self.INCLUDE_SUB_FOLDERS = [f'{self.folder_name}/PanelA']

        col_number = 2

        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "C9": {
                    "Untreated": [(row_rep1, col_number), (row_rep2, col_number), (row_rep3, col_number)]
                }
            },
            
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3", "ch4"],
            self.KEY_MARKERS: {
                'panelA': ["Vimentin", "WDR49", "DAPI", "PML"]
                },
            self.KEY_REPS: ["rep1", "rep2", "rep3"],
        }

#######################################


class Config_C9_PanelB(Config_C9):
    def __init__(self):
        super().__init__()

        self.INCLUDE_SUB_FOLDERS = [f'{self.folder_name}/PanelB']

        col_number = 3

        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "C9": {
                    "Untreated": [(row_rep1, col_number), (row_rep2, col_number), (row_rep3, col_number)]
                }
            },
            
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3", "ch4"],
            self.KEY_MARKERS: {
                'panelB': ["ARL13B", "WDR49", "DAPI", "Vimentin"]
                },
            self.KEY_REPS: ["rep1", "rep2", "rep3"],
        }

#######################################

class Config_C9_PanelC(Config_C9):
    def __init__(self):
        super().__init__()

        self.INCLUDE_SUB_FOLDERS = [f'{self.folder_name}/PanelC']

        col_number = 6

        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "C9": {
                    "Untreated": [(row_rep1, col_number), (row_rep2, col_number), (row_rep3, col_number)]
                }
            },
            
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3", "ch4"],
            self.KEY_MARKERS: {
                'panelC': ["Vimentin", "Calreticulin", "DAPI", "PML"]
                },
            self.KEY_REPS: ["rep1", "rep2", "rep3"],
        }

#######################################

class Config_C9_PanelD(Config_C9):
    def __init__(self):
        super().__init__()

        self.INCLUDE_SUB_FOLDERS = [f'{self.folder_name}/PanelD']

        col_number = 7

        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "C9": {
                    "Untreated": [(row_rep1, col_number), (row_rep2, col_number), (row_rep3, col_number)]
                }
            },
            
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3", "ch4"],
            self.KEY_MARKERS: {
                'panelD': ["Vimentin", "WDR49", "DAPI", "TDP43"]
                },
            self.KEY_REPS: ["rep1", "rep2", "rep3"],
        }

#######################################