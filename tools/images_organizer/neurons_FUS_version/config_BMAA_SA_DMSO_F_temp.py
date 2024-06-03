######################################################
########## Please Don't Change This Section ##########
######################################################

import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from tools.images_organizer.neurons_FUS_version.config import Config

class Config_BSD_F(Config):
    def __init__(self):
        super().__init__()

        # What we already ran:
        self.FOLDERS = ['20242801_BMAA_SA_DMSO_4d_D-F']

        self.EXCLUDE_SUB_FOLDERS = ['20242801_BMAA_SA_DMSO_4d_D-F/PanelD', '20242801_BMAA_SA_DMSO_4d_D-F/PanelE']
        self.RAISE_ON_MISSING_INDEX = False

        self.FILENAME_POSTFIX = "BSD_"
        ##################################

        ########################################
        ############### Advanced ###############
        ########################################

        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "KOLF": {
                    "DMSO": [(4001,4100),(4101,4200)],
                    "Untreated": [(5501, 5600), (5401, 5500)]
                },
                "FUSHeterozygous": {
                    "DMSO": [(4301,4400),(4201,4300)],
                },
                "FUSRevertant": {
                    "SA": [(3101,3200),(3001,3100)],
                    "DMSO": [(5201,5300),(5301,5400)]
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