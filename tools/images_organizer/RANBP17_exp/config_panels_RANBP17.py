######################################################
########## Please Don't Change This Section ##########
######################################################

import os
import sys
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))

from tools.images_organizer.RANBP17_exp.config import Config
import pandas as pd



mapping = {
    "iW11": {
        "tardp_kd": [(2,4), (3,4)],
        "ranbp17_kd": [(4,4), (5,4)],
        "control_179": [(2,10), (3,10)],
        "control_180": [(4,10), (5,10)],
        "both_kd": [(2,16), (3,16)],
        "untreated": [(4,18),(5,18),(4,19), (5,19)],
    }
}


class Config_Base_Data(Config):
    def __init__(self, plate):
        super().__init__()



class Config_Base_3Markers(Config_Base_Data):
    def __init__(self, plate):
        super().__init__(plate)
        self.plate = plate
        self.FOLDERS = [f"plate{self.plate}"]
        self.INCLUDE_SUB_FOLDERS = [f'plate{self.plate}/panel{self.panel}']
        self.CONFIG = {
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3"],
            self.KEY_REPS: ["rep1", "rep2", "rep3", "rep4",],
            self.KEY_CELL_LINES: mapping,
            self.KEY_MARKERS: {f'panel{self.panel}': self.markers}
        }


class Config_A(Config_Base_3Markers):
    def __init__(self, plate):
        # Params:
        self.col_rep1 = 1
        self.col_rep2 = 2
        self.panel = 'A'
        self.markers = ["RANBP17", "DAPI", "TDP-43"]
        
        super().__init__(plate)



if __name__ == "__main__":

    print(mapping)
        


