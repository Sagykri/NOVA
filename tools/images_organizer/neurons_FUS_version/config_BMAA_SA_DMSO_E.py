######################################################
########## Please Don't Change This Section ##########
######################################################

import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from tools.images_organizer.neurons_FUS_version.config import Config

class Config_BSD_E(Config):
    def __init__(self):
        super().__init__()

        # What we already ran:
        self.FOLDERS = ['20242801_BMAA_SA_DMSO_4d_D-F']

        self.EXCLUDE_SUB_FOLDERS = ['20242801_BMAA_SA_DMSO_4d_D-F/PanelD', '20242801_BMAA_SA_DMSO_4d_D-F/PanelF']


        self.FILENAME_POSTFIX = "BSD_"
        ##################################

        ########################################
        ############### Advanced ###############
        ########################################

        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "KOLF": {
                    "BMAA": [(201,300),(301,400)],
                    "SA": [(2101,2200),(2001,2100)],
                    "DMSO": [(3901,4000),(3801,3900)],
                    "Untreated": [(5601, 5700), (5701, 5800)]
                },
                "FUS_Heterozygous": {
                    "BMAA": [(901,1000),(801,900)],
                    "SA": [(2601,2700),(2701,2800)],
                    "DMSO": [(4401,4500),(4501,4600)],
                },
                "FUS_Revertant": {
                    "BMAA": [(1401,1500),(1501,1600)],
                    "SA": [(3201,3300),(3301,3400)],
                    "DMSO": [(5101,5200),(5001,5100)]
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