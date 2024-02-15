######################################################
########## Please Don't Change This Section ##########
######################################################

import os
import sys

sys.path.insert(1, os.getenv("MOMAPS_HOME"))
from tools.images_organizer.neurons_FUS_version.config_BMAA_SA_DMSO import Config_BSD

class Config_BSD_G_I(Config_BSD):
    def __init__(self):
        super().__init__()

        # What we already ran:
        self.FOLDERS = ['20242901_BMAA_SA_DMSO_4d_G-I']

        # self.EXCLUDE_SUB_FOLDERS = ['20242401_BMAA_SA_DMSO_4d_A-C/PanelB', '20242401_BMAA_SA_DMSO_4d_A-C/PanelC']

#######################################