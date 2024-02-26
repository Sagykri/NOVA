######################################################
########## Please Don't Change This Section ##########
######################################################

import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from tools.images_organizer.neurons_FUS_version.config import Config

class Config_CCN_G_I(Config):
    def __init__(self):
        super().__init__()
        

        self.FOLDERS = ['20240205_Colchicine_Cisplatin_NMS873_2d1d_G-I']
        # self.EXCLUDE_SUB_FOLDERS = ["20240204_Colchicine_Cisplatin_NMS873_2d1d_D-F/PanelD", "20240204_Colchicine_Cisplatin_NMS873_2d1d_D-F/PanelE"]


        self.FILENAME_POSTFIX = "CCN_"

        ##################################

        ########################################
        ############### Advanced ###############
        ########################################

        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "KOLF": {
                    "Colchicine": [(1,100),(101,200)],
                    "Cisplatin": [(701,800),(601,700)],
                    "NMS873": [(1201,1300),(1301,1400)],
                    "Untreated": [(1901, 2000), (1801, 1900)]
                },
                "FUSHeterozygous": {
                    "Colchicine": [(301,400),(201,300)],
                    "Cisplatin": [(801,900),(901,1000)],
                    "NMS873": [(1501,1600),(1401,1500)]
                },
                "FUSRevertant": {
                    "Colchicine": [(401,500),(501,600)],
                    "Cisplatin": [(1101,1200),(1001,1100)],
                    "NMS873": [(1601,1700),(1701,1800)]
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