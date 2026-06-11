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

mapping_plate3_cortical_day12_partial = {
    "iW11": {
        "tardp-kd": [(2,2), (3,2)],
        "ranbp17-kd": [(2,3), (3,3)],
        "both-kd": [(2,4), (3,4)],
        "control-179": [(2,5), (3,5)],
        "control-180": [(2,6), (3,6)],
        "untreated": [(2,7),(3,7)],
    }
}

mapping_plate3_cortical_day12_full = {
    "iW11": {
        "tardp-kd": [(2,2), (3,2), (4,2), (5,2), (6,2), (7,2), (8,2), (9,2)],
        "ranbp17-kd": [(2,3), (3,3), (4,3), (5,3), (6,3), (7,3), (8,3), (9,3)],
        "both-kd": [(2,4), (3,4), (4,4), (5,4), (6,4), (7,4), (8,4), (9,4)],
        "control-179": [(2,5), (3,5), (4,5), (5,5), (6,5), (7,5), (8,5), (9,5)],
        "control-180": [(2,6), (3,6), (4,6), (5,6), (6,6), (7,6), (8,6), (9,6)],
        "untreated": [(2,7),(3,7), (4,7), (5,7), (6,7), (7,7), (8,7), (9,7)],
    }
}

mapping_plate3_cortical_day12_full_panelB = {
    "iW11": {
        "tardp-kd": [(1,2), (10, 2), (11,2)],
        "ranbp17-kd": [(1,3), (10, 3), (11,3)],
        "both-kd": [(1,4), (10, 4), (11,4)],
        "control-179": [(1,5), (10, 5), (11,5)],
        "control-180": [(1,6), (10, 6), (11,6)],
        "untreated": [(1,7),(10,7), (11,7)]
    }
}

class Config_Base_Data(Config):
    def __init__(self, plate):
        super().__init__()


def get_mapping(plate_key):
    if plate_key == "1A":
        return mapping_plate1_cortical_day8
    elif plate_key == "2A":
        return mapping_plate2_motor_day8
    elif plate_key == "3A":
        return mapping_plate3_cortical_day12_full
    elif plate_key == "3B":
        return mapping_plate3_cortical_day12_full_panelB
    else:
        raise ValueError(f"Invalid plate number - no mapping available for plate: {plate_key}")

def get_max_reps(mapping):
    max_reps = 0
    for cell_line, cond_dict in mapping.items():
        for condition, wells in cond_dict.items():
            num_wells = len(wells)
            if num_wells > max_reps:
                max_reps = num_wells
    return max_reps

class Config_Base_3Markers(Config_Base_Data):
    def __init__(self, plate):
        super().__init__(plate)
        self.plate = plate
        self.FOLDERS = [f"plate{self.plate}"]
        self.INCLUDE_SUB_FOLDERS = [f'plate{self.plate}/panel{self.panel}']
        mapping = get_mapping(self.mapping_key)
        self.CONFIG = {
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3"],
            self.KEY_REPS: [f"rep{rep}" for rep in range(1,get_max_reps(mapping) + 1)],
            self.KEY_CELL_LINES: mapping.copy(),
            self.KEY_MARKERS: {f'panel{self.panel_alias}': self.markers}
        }


class Config_A(Config_Base_3Markers):
    def __init__(self, plate):
        # Params:
        self.panel = 'A'
        self.panel_alias = 'A'
        self.markers = ["RANBP17", "DAPI", "TDP-43"]
        self.mapping_key = f"{plate}{self.panel}"
        
        super().__init__(plate)


class Config_B(Config_Base_3Markers):
    def __init__(self, plate):
        # Params:
        self.panel = 'A' # Same input folder
        self.panel_alias = 'B' # different output folder and markers
        self.markers = ["RANBP17_SG", "DAPI", "SG"]
        self.mapping_key = f"{plate}{self.panel_alias}"
        super().__init__(plate)



if __name__ == "__main__":
    cf = Config_B(3)   
    print(cf.CONFIG)

