######################################################
########## Please Don't Change This Section ##########
######################################################

import os
import sys
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))

from tools.images_organizer.RANBP17_exp.config import Config
import pandas as pd



mapping_plate1_cortical_day8 = {
    "iW11": {
        "tardp-kd": [(2,4), (3,4)],
        "ranbp17-kd": [(4,4), (5,4)],
        "control-179": [(2,10), (3,10)],
        "control-180": [(4,10), (5,10)],
        "both-kd": [(2,16), (3,16)],
        "untreated": [(4,18),(5,18),(4,19), (5,19)],
    }
}

mapping_plate2_motor_day8 = {
    "iW11": {
        "tardp-kd": [(3,2), (3,3), (4,2), (4,3)],
        "ranbp17-kd":  [(5,2), (5,3), (6,2), (6,3)],
        "both-kd":  [(3,14), (3,15), (4,14), (4,15)],
        "control-179":  [(3,8), (3,9), (4,8), (4,9)],
        "control-180":  [(5,8), (5,9), (6,8), (6,9)],
    }
}

mapping_plate3_cortical_day12 = {
    "iW11": {
        "tardp-kd": [(2,2), (3,2)],
        "ranbp17-kd": [(2,3), (3,3)],
        "both-kd": [(2,4), (3,4)],
        "control-179": [(2,5), (3,5)],
        "control-180": [(2,6), (3,6)],
        "untreated": [(2,7),(3,7)],
    }
}

class Config_Base_Data(Config):
    def __init__(self, plate):
        super().__init__()


def get_mapping(plate):
    if plate == 1:
        return mapping_plate1_cortical_day8
    elif plate == 2:
        return mapping_plate2_motor_day8
    elif plate == 3:
        return mapping_plate3_cortical_day12
    else:
        raise ValueError(f"Invalid plate number - no mapping available for plate: {plate}")

class Config_Base_3Markers(Config_Base_Data):
    def __init__(self, plate):
        super().__init__(plate)
        self.plate = plate
        self.FOLDERS = [f"plate{self.plate}"]
        self.INCLUDE_SUB_FOLDERS = [f'plate{self.plate}/panel{self.panel}']
        self.CONFIG = {
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3"],
            self.KEY_REPS: ["rep1", "rep2", "rep3", "rep4",],
            self.KEY_CELL_LINES: get_mapping(self.plate),
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



        


