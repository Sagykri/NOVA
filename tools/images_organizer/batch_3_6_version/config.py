######################################################
########## Please Don't Change This Section ##########
######################################################

import os

LOGGING_PATH = "log.log"
KEY_CELL_LINES = "cell_lines"
KEY_MARKERS_ALIAS_ORDERED = "markers_alias_ordered"
KEY_MARKERS = "markers"
KEY_BATCHES = "batches"
KEY_REPS = "reps"
FILE_EXTENSION = ".tif"

#####################################################################



# You may change the configuration beneath this line


#####################################
############### Paths ###############
#####################################

# Path to source folder (root)
SRC_ROOT_PATH = os.path.join('..', 'input_example')

# Path to destination folder (root)
DST_ROOT_PATH = os.path.join('..', 'output')

# Names of folders to handle
# - For selecting all folders in SRC_ROOT_PATH - set FOLDERS to None or delete the assignment 
# for example:
#           FOLDERS = None
# - For selecting specific folders in SRC_ROOT_PATH - set FOLDERS to array of folders names.
# for example:
# FOLDERS = ['230220_plate1.2_rowB_panelD', '230222_plate2.2_rowB_panelD',
#           '230224_plate3.2_rowB_panelC', '230222_plate2.2_rowB_panelD',
#           '230226_plate1.2_rowC_panelD', '230301_plate3.2_rowC_panelD']
FOLDERS = None

##################################

########################################
############### Advanced ###############
########################################

CONFIG = {
    KEY_CELL_LINES: {
        "WT": {
            "stress": (1,100),
            "Untreated": (101,200)
        },
        "TDP43": {
            "Untreated": (201,300)
        },
        "OPTN": {
            "Untreated": (301, 400)
        },
        "FUSHomozygous": {
            "Untreated": (401,500)
        },
        "TBK1": {
            "Untreated": (501,600)
        },
        "FUSHeterozygous": {
            "Untreated": (601,700)
        },
        "FUSRevertant": {
            "Untreated": (701,800)
        },
        "SCNA": {
            "Untreated": (801,900)
        }
    },
    KEY_MARKERS_ALIAS_ORDERED: ["Cy5", "mCherry", "GFP", "DAPI"],
    KEY_MARKERS: {
        "panelA": ["G3BP1", "KIF5A", "PURA", "DAPI"],
        "panelB": ["NONO", "TDP43", "CD41", "DAPI"],
        "panelC": ["SQSTM1", "FMRP", "Phalloidin", "DAPI"],
        "panelD": ["PSD95", "CLTC", None, "DAPI"],
        "panelE": ["NEMO", "DCP1A", None, "DAPI"],
        "panelF": ["GM130", "TOMM20", None, "DAPI"],
        "panelG": ["NCL", "FUS", None, "DAPI"],
        "panelH": ["ANXA11", "SCNA", None, "DAPI"],
        "panelI": ["Calreticulin", "LAMP1", None, "DAPI"],
        "panelJ": [None, "TIA1", None, "DAPI"],
        "panelK": ["mitotracker", "PML", "PEX14", "DAPI"]
    },
    KEY_BATCHES: {
        "batch3": ["plate1.1", "plate1.2", "plate1.3", "plate1.4"],
        "batch4": ["plate2.1", "plate2.2", "plate2.3", "plate2.4"],
        "batch5": ["plate3.1", "plate3.2", "plate3.3", "plate3.4"]
    },
    KEY_REPS: {
        "rep1": ["rowB", "rowD", "rowF"],
        "rep2": ["rowC", "rowE", "rowG"]
    }
}

#######################################