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
SRC_ROOT_PATH = "/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/SpinningDisk/batches6_9"

# Path to destination folder (root)
DST_ROOT_PATH = "/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/SpinningDisk"

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

# If set to False, the files will be *copied* to DST_ROOT_PATH, otherwise, the files will be *cut*/*moved* to DST_ROOT_PATH
CUT_FILES = True

##################################

########################################
############### Advanced ###############
########################################

CONFIG = {
    KEY_CELL_LINES: {
        "WT": {
            "stress": [(1,100),(101,200)],
            "Untreated": [(201,300), (301,400)]
        },
        "TDP43": {
            "Untreated": [(401,500), (501, 600)]
        },
        "OPTN": {
            "Untreated": [(601, 700), (701,800)]
        },
        "FUSHomozygous": {
            "Untreated": [(801,900),(901,1000)]
        },
        "TBK1": {
            "Untreated": [(1001,1100), (1101,1200)]
        },
        "FUSHeterozygous": {
            "Untreated": [(1201,1300), (1301,1400)]
        },
        "FUSRevertant": {
            "Untreated": [(1401,1500), (1501,1600)]
        },
        "SCNA": {
            "Untreated": [(1601,1700), (1701,1800)]
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
        "batch6": ["plate1.1", "plate1.2", "plate1.3", "plate1.4"],
        "batch7": ["plate2.1", "plate2.2", "plate2.3", "plate2.4"],
        "batch8": ["plate3.1", "plate3.2", "plate3.3", "plate3.4"],
        "batch9": ["plate4.1", "plate4.2", "plate4.3", "plate4.4"]
    },
    KEY_REPS: ["rep1", "rep2"]
}

#######################################