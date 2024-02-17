######################################################
########## Please Don't Change This Section ##########
######################################################

import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from tools.images_organizer.neurons_FUS_version.config import Config

class Config_MME(Config):
    def __init__(self):
        super().__init__()

        # What we already ran:
        self.FOLDERS = ['20243001_MG132_ML240_Etoposide_4d_A-C']
        # self.EXCLUDE_SUB_FOLDERS = ['20243001_MG132_ML240_Etoposide_4d_A-C/PanelB', '20243001_MG132_ML240_Etoposide_4d_A-C/PanelC']


        self.FILENAME_POSTFIX = "MME_"
        ##################################

        ########################################
        ############### Advanced ###############
        ########################################

        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "KOLF": {
                    "MG132": [(1,100),(101,200)],
                    "ML240": [(701,800),(601,700)],
                    "Etoposide": [(1201,1300),(1301,1400)],
                    "Untreated": [(1901, 2000), (1801, 1900)]
                },
                "FUSHeterozygous": {
                    "MG132": [(301,400),(201,300)],
                    "ML240": [(801,900),(901,1000)],
                    "Etoposide": [(1501,1600),(1401,1500)],
                },
                "FUSRevertant": {
                    "MG132": [(401,500),(501,600)],
                    "ML240": [(1101,1200),(1001,1100)],
                    "Etoposide": [(1601,1700),(1701,1800)],
                }
            },
            self.KEY_MARKERS_ALIAS_ORDERED: ["DAPI", "GFP", "mCherry", "Cy5"],
            self.KEY_MARKERS: {
                "panelA": ["DAPI", "PURA", "G3BP1", "KIF5A"],
                "panelB": ['DAPI', 'CD41', 'NONO','TDP43'],
                "panelC": ['DAPI', "Phalloidin", "SQSTM1", "FMRP"],
                "panelD": ['DAPI', "PSD95", None, "CLTC"],
                "panelE": ['DAPI', "NEMO",None, "DCP1A"],
                "panelF": ['DAPI', "GM130", None, "TOMM20"],
                "panelG": ['DAPI', "FUS", None, "NCL"],
                "panelH": ['DAPI', "SNCA", None, "ANXA11"],
                "panelI": ['DAPI', "LAMP1", None, "Calreticulin"],
                "panelJ": ['DAPI', "PEX14", "PML", "mitotracker"],
                },
                self.KEY_REPS: ["rep1", "rep2"],

        }

#######################################