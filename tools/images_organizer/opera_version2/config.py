######################################################
########## Please Don't Change This Section ##########
######################################################

import os

class Config():
    def __init__(self):
        super().__init__()

        self.LOGGING_PATH = os.path.join('tools', 'images_organizer', 'opera_version2', 'logs', 'iAstrocytes') #os.path.join('tools', 'images_organizer', 'opera_version2', 'logs', 'Coyne080525') #
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
        # self.SRC_ROOT_PATH = "/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/Opera18DaysReimaged/"
        self.SRC_ROOT_PATH = "/home/projects/hornsteinlab/Collaboration/MOmaps/input/images/raw/John/iAstrocytes/pre-ordered/"
        # self.SRC_ROOT_PATH = "/home/projects/hornsteinlab/Collaboration/MOmaps/input/images/raw/AlyssaCoyne/Coyne_080525/"

        # Path to destination folder (root)
        # self.DST_ROOT_PATH = "/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/Opera18DaysReimaged_sorted/"
        self.DST_ROOT_PATH = "/home/projects/hornsteinlab/Collaboration/MOmaps/input/images/raw/John/iAstrocytes/ordered/"
        # self.DST_ROOT_PATH = "/home/projects/hornsteinlab/Collaboration/MOmaps/input/images/raw/AlyssaCoyne/Coyne_080525_ordered/"

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
        self.INCLUDE_SUB_FOLDERS = []

        # If set to False, the files will be *copied* to DST_ROOT_PATH, otherwise, the files will be *cut*/*moved* to DST_ROOT_PATH
        self.CUT_FILES = False
        
        # Raise exception when index couldn't be found in the config?
        self.RAISE_ON_MISSING_INDEX = True

        self.FILENAME_POSTFIX = ""
        ##################################
        
        #######################################