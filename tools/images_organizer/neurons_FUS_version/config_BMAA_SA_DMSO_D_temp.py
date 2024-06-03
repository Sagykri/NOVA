######################################################
########## Please Don't Change This Section ##########
######################################################

import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from tools.images_organizer.neurons_FUS_version.config import Config

class Config_BSD_D(Config):
    def __init__(self):
        super().__init__()

        # What we already ran:
        self.FOLDERS = ['20242801_BMAA_SA_DMSO_4d_D-F']

        self.EXCLUDE_SUB_FOLDERS = ['20242801_BMAA_SA_DMSO_4d_D-F/PanelE', '20242801_BMAA_SA_DMSO_4d_D-F/PanelF']
        self.RAISE_ON_MISSING_INDEX = False

        self.FILENAME_POSTFIX = "BSD_"
        ##################################

        ########################################
        ############### Advanced ###############
        ########################################

        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "KOLF": {
                    "DMSO": [(3601,3700),(3701,3800)],
                    "Untreated": [(5901, 6000), (5801, 5900)]
                },
                "FUSHeterozygous": {
                    "DMSO": [(4701,4800),(4601,4700)],
                },
                "FUSRevertant": {
                    "SA": [(3501,3600),(3401,3500)],
                    "DMSO": [(4801,4900),(4901,5000)]
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