######################################################
########## Please Don't Change This Section ##########
######################################################

import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from tools.images_organizer.opera_version.config import Config

class Config_G(Config):
    def __init__(self):
        super().__init__()

        folder_name = "batch1"
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
                "KOLF": {
                    "BMAA": [(row_rep1,2), (row_rep2,2)],
                    "SA": [(row_rep1,5), (row_rep2,5)],
                    "DMSO": [(row_rep1,8), (row_rep2,8)],
                    "Untreated": [(row_rep1,11), (row_rep2,11)]
                },
                "FUSHeterozygous": {
                    "BMAA": [(row_rep1,3), (row_rep2,3)],
                    "SA": [(row_rep1,6), (row_rep2,6)],
                    "DMSO": [(row_rep1,9), (row_rep2,9)],
                },
                "FUSRevertant": {
                    "BMAA": [(row_rep1,4), (row_rep2,4)],
                    "SA": [(row_rep1,7), (row_rep2,7)],
                    "DMSO": [(row_rep1,10), (row_rep2,10)],
                }
            },
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch3", "ch2"],
            # [DAPI, GFP, Cy5]
            # ch1 - DAPI
            # ch2 - Cy5
            # ch3 - GFP
            self.KEY_MARKERS: {
                "panelG": ['DAPI', "FUS", "NCL"],
                "panelH": ['DAPI', "SNCA", "ANXA11"],
                "panelI": ['DAPI', "LAMP1", "Calreticulin"]
                },
                self.KEY_REPS: ["rep1", "rep2"],
        }

#######################################