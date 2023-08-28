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
SRC_ROOT_PATH = "/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/SpinningDisk/microglia_LPS"

# Path to destination folder (root)
DST_ROOT_PATH = "/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/SpinningDisk/microglia_LPS_sort"

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
        "WT": {
            "Untreated": [(1,100),(101,200)],
            "LPS":[(1001,1100),(1101,1200)]
        },
        "TDP43": {
            "Untreated": [(201,300), (301, 400)],
            "LPS":[(1201,1300),(1301,1400)]
        },
        "OPTN": {
            "Untreated": [(401, 500), (501,600)],
            "LPS":[(1401,1500),(1501,1600)]
        },
        "FUSHomozygous": {
            "Untreated": [(601,700),(701,800)],
            "LPS":[(1601,1700),(1701,1800)]
        },
        "TBK1": {
            "Untreated": [(801,900), (901,1000)],
            "LPS":[(1801,1900),(1901,2000)]
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
        "panelG": ["FUS", "NCL", None, "DAPI"],
        "panelH": ["SCNA", "ANXA11", None, "DAPI"],
        "panelI": ["LAMP1", "Calreticulin", None, "DAPI"],
        "panelJ": ["TIA1", "pNFKB", None, "DAPI"],
        "panelK": [None, "PML", "PEX14", "DAPI"]
        },
        KEY_REPS: ["rep1", "rep2"],
        KEY_COL_WELLS: {
            "2": ['WT','Untreated'],
            "3": ['TDP43','Untreated'],
            "4": ['OPTN','Untreated'],
            "5": ['FUSHomozygous','Untreated'],
            "6": ['TBK1','Untreated'],
            "7": ['WT','LPS'],
            "8": ['TDP43','LPS'],
            "9": ['OPTN','LPS'],
            "10": ['FUSHomozygous','LPS'],
            "11": ['TBK1','LPS'],
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