######################################################
########## Please Don't Change This Section ##########
######################################################

import os

class Config():
    def __init__(self):
        super().__init__()

        self.LOGGING_PATH = "logs"
        self.KEY_CELL_LINES = "cell_lines"
        self.KEY_MARKERS_ALIAS_ORDERED = "markers_alias_ordered"
        self.KEY_MARKERS = "markers"
        self.KEY_BATCHES = "batches"
        self.KEY_REPS = "reps"
        self.FILE_EXTENSION = ".tif"
        self.KEY_COL_WELLS = "wells_columns"
        self.KEY_ROW_WELLS = "wells_rows"

        #####################################################################

        # You may change the configuration beneath this line


        #####################################
        ############### Paths ###############
        #####################################

        # Path to source folder (root)
        self.SRC_ROOT_PATH = "/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/FUS_lines_stress_2024_unordered"

        # Path to destination folder (root)
        self.DST_ROOT_PATH = "/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/FUS_lines_stress_2024_sorted"

        # Names of folders to handle
        # - For selecting all folders in SRC_ROOT_PATH - set FOLDERS to None or delete the assignment 
        # for example:
        #           FOLDERS = None
        # - For selecting specific folders in SRC_ROOT_PATH - set FOLDERS to array of folders names.
        # for example:
        # FOLDERS = ['230220_plate1.2_rowB_panelD', '230222_plate2.2_rowB_panelD',
        #           '230224_plate3.2_rowB_panelC', '230222_plate2.2_rowB_panelD',
        #           '230226_plate1.2_rowC_panelD', '230301_plate3.2_rowC_panelD']

        # What we already ran:
        self.FOLDERS = []
        # Running now:
        # FOLDERS = ['20243001_MG132_ML240_Etoposide_4d_D-F']
        # FOLDERS = ['20230202_MG132_ML240_Etoposide_4d_J']
        # FOLDERS = ['20243001_MG132_ML240_Etoposide_4d_G-I']
        # EXCLUDE_SUB_FOLDERS = ['20243001_MG132_ML240_Etoposide_4d_G-I/PanelG']

        self.EXCLUDE_SUB_FOLDERS = []

        # If set to False, the files will be *copied* to DST_ROOT_PATH, otherwise, the files will be *cut*/*moved* to DST_ROOT_PATH
        self.CUT_FILES = False

        self.FILENAME_POSTFIX = ""
        ##################################

        ########################################
        ############### Advanced ###############
        ########################################

        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "KOLF": {
                    "MG132": [(1,100),(101,200)],
                    "ML240": [(701,800),(601,700)],
                    "Etoposide": [(1201,1300),(1301,1400)],
                    "Untreated": [(1901, 2000), (1801, 1900)]
                },
                "FUS_Heterozygous": {
                    "MG132": [(301,400),(201,300)],
                    "ML240": [(801,900),(901,1000)],
                    "Etoposide": [(1501,1600),(1401,1500)],
                },
                "FUS_Revertant": {
                    "MG132": [(401,500),(501,600)],
                    "ML240": [(1101,1200),(1001,1100)],
                    "Etoposide": [(1601,1700),(1701,1800)],
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