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


        self.FILENAME_POSTFIX = "BSD_"
        ##################################

        ########################################
        ############### Advanced ###############
        ########################################

        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "KOLF": {
                    "BMAA": [(1,100),(101,200)],
                    "SA": [(2301,2400),(2201,2300)],
                    "DMSO": [(4101,4200),(4001,4100)],
                    "Untreated": [(5401, 5500), (5501, 5600)]
                },
                "FUS_Heterozygous": {
                    "BMAA": [(1101,1200),(1001,1100)],
                    "SA": [(2401,2500),(2501,2600)],
                    "DMSO": [(4201,4300),(4301,4400)],
                },
                "FUS_Revertant": {
                    "BMAA": [(1201,1300),(1301,1400)],
                    "SA": [(3001,3100),(3101,3200)],
                    "DMSO": [(5301,5400),(5201,5300)]
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