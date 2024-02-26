######################################################
########## Please Don't Change This Section ##########
######################################################

import os
import sys
import os

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from tools.images_organizer.neurons_FUS_version.config_MG132_ML240_Etoposide import Config_MME

class Config_MME_G(Config_MME):
    def __init__(self):
        super().__init__()

        # What we already ran:
        self.FOLDERS = ['20243001_MG132_ML240_Etoposide_4d_G-I']
        self.EXCLUDE_SUB_FOLDERS = ['20243001_MG132_ML240_Etoposide_4d_G-I/PanelH', '20243001_MG132_ML240_Etoposide_4d_G-I/PanelI']
        
#######################################