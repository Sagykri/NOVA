######################################################
########## Please Don't Change This Section ##########
######################################################
import os
import sys
sys.path.insert(0, os.getenv("HOME"))
sys.path.insert(1, os.getenv("NOVA_HOME"))
print(f"NOVA_HOME: {os.getenv('NOVA_HOME')}")

FUNOVA_DATA_DIR = "/home/projects/hornsteinlab/Collaboration/Guy_Lior/fuNOVA_Pilot"
class Config():
    def __init__(self):
        super().__init__()
        
        self.LOGGING_PATH = os.path.join(os.getenv("NOVA_LOCAL"), 'tools', 'images_organizer', 'FuNOVA_Guy_Lior', 'logs', 'pilot')
        self.KEY_CELL_LINES = "cell_lines"
        self.KEY_MARKERS_ALIAS_ORDERED = "markers_alias_ordered"
        self.KEY_MARKERS = "markers"
        self.KEY_BATCHES = "batches"
        self.KEY_REPS = "reps"
        self.FILE_EXTENSION = ".tiff"
        self.KEY_COL_WELLS = "wells_columns"
        self.KEY_ROW_WELLS = "wells_rows"

        #####################################################################

        # You may change the configuration beneath this line


        #####################################
        ############### Paths ###############
        #####################################

        # Path to source folder (root)
        self.SRC_ROOT_PATH = os.path.join(FUNOVA_DATA_DIR, "zstack_collapse_2nd_imaging")

        # Path to destination folder (root)
        self.DST_ROOT_PATH = os.path.join(FUNOVA_DATA_DIR, "zstack_collapse_2nd_imaging_sorted")

        # Names of folders to handle
        # - For selecting all folders in SRC_ROOT_PATH - set FOLDERS to None or delete the assignment 
        # for example:
        #           FOLDERS = None
        # - For selecting specific folders in SRC_ROOT_PATH - set FOLDERS to array of folders names.
        # for example:
        # FOLDERS = ['230220_plate1.2_rowB_panelD', '230222_plate2.2_rowB_panelD',
        #           '230224_plate3.2_rowB_panelC', '230222_plate2.2_rowB_panelD',
        #           '230226_plate1.2_rowC_panelD', '230301_plate3.2_rowC_panelD']

        self.FOLDERS = []
        self.EXCLUDE_SUB_FOLDERS = []
        self.INCLUDE_SUB_FOLDERS = []

        # If set to True, running as "dry run" - doesn't copy/move files, only prints to logs
        self.DRY_RUN = False
        
        # If set to False, the files will be *copied* to DST_ROOT_PATH, otherwise, the files will be *cut*/*moved* to DST_ROOT_PATH
        self.CUT_FILES = False
        
        # Raise exception when index couldn't be found in the config?
        self.RAISE_ON_MISSING_INDEX = True

        self.FILENAME_POSTFIX = ""
        ##################################
        
        #######################################