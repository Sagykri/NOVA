######################################################
########## Please Don't Change This Section ##########
######################################################

import os

LOGGING_PATH = "log"
KEY_CELL_LINES = "cell_lines"
KEY_MARKERS_ALIAS_ORDERED = "markers_alias_ordered"
KEY_MARKERS = "markers"
KEY_BATCHES = "batches"
KEY_REPS = "reps"
FILE_EXTENSION = ".tif"
KEY_COL_WELLS = "wells_columns"
KEY_ROW_WELLS = "wells_rows"

#####################################################################



# You may change the configuration beneath this line


#####################################
############### Paths ###############
#####################################

# Path to source folder (root)
SRC_ROOT_PATH = "/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/SpinningDisk/deltaNLS"

# Path to destination folder (root)
DST_ROOT_PATH = "/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/SpinningDisk/deltaNLS_sort"

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
CUT_FILES = False

##################################

########################################
############### Advanced ###############
########################################

CONFIG = {
    KEY_CELL_LINES: {
        "TDP43": {
            "Untreated": [(1,100),(101,200)],
            "dox": [(201,300), (301,400)]
        },
        "WT": {
            "Untreated": [(401,500), (501, 600)]}
    },
    KEY_MARKERS_ALIAS_ORDERED: ["Cy5", "mCherry", "GFP", "DAPI"],
    KEY_MARKERS: {
        "panelA": ["G3BP1", "KIF5A", "PURA", "DAPI"],
        "panelB": ["TDP43", "DCP1A", "Tubulin", "DAPI"],
        "panelC": ["SQSTM1", "FMRP", "Phalloidin", "DAPI"],
        "panelD": ["PSD95", "CLTC", "CD41", "DAPI"],
        "panelE": [None, "KIF20A", None, "DAPI"],
        "panelF": ["GM130", "TOMM20", None, "DAPI"],
        "panelG": ["NCL", "FUS", None, "DAPI"],
        "panelH": ["ANXA11", "SCNA", None, "DAPI"],
        "panelI": ["Calreticulin", "LAMP1", None, "DAPI"],
        "panelJ": ["Pericentrin", "TIA1", None, "DAPI"],
        "panelK": ["Rab5", "NONO", None, "DAPI"],
        "panelL": ["KIFC1", "NEMO", None, "DAPI"],
        "panelM": ["mitotracker", "PML", "PEX14", "DAPI"],
        "panelN": [None, "TDP43", None, "DAPI"],
        },
        KEY_REPS: ["rep1", "rep2"],
        KEY_COL_WELLS: {
            "2": ['TDP43','Untreated'],
            "5": ['TDP43','Untreated'],
            "8": ['TDP43','Untreated'],
            "3": ['TDP43','dox'],
            "6": ['TDP43','dox'],
            "9": ['TDP43','dox'],
            "4": ['WT','Untreated'],
            "7": ['WT','Untreated'],
            "10": ['WT','Untreated']
        },
        KEY_ROW_WELLS :{
            "B": 'rep1',
            "D": 'rep1',
            "F": 'rep1',
            "C": 'rep2',
            "E": 'rep2',
            "G": 'rep2'
        }
}

#######################################