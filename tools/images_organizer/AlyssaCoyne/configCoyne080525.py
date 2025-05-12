######################################################
########## Please Don't Change This Section ##########
######################################################

import os

class Config():
    def __init__(self):
        super().__init__()
        
        self.LOGGING_PATH = os.path.join('tools', 'images_organizer', 'AlyssaCoyne', 'logs', 'Coyne080525') #os.path.join('tools', 'images_organizer', 'opera_version2', 'logs', 'iAstrocytes')
        self.KEY_CELL_LINES = "cell_lines"
        self.KEY_MARKERS_ALIAS_ORDERED = "markers_alias_ordered"
        self.KEY_MARKERS = "markers"
        self.KEY_BATCHES = "batches"
        self.KEY_REPS = "reps"
        self.FILE_EXTENSION = ".tif"
        self.KEY_COL_WELLS = "wells_columns"
        self.KEY_ROW_WELLS = "wells_rows"

        #####################################
        ############### Paths ###############
        #####################################

        # Path to source folder (root)
        self.SRC_ROOT_PATH = "/home/projects/hornsteinlab/Collaboration/MOmaps/input/images/raw/AlyssaCoyne/Coyne_080525"
        # Path to destination folder (root)
        self.DST_ROOT_PATH = "/home/projects/hornsteinlab/Collaboration/MOmaps/input/images/raw/AlyssaCoyne/Coyne_080525_sorted"

        # Names of folders to handle
        self.FOLDERS = []


        self.EXCLUDE_SUB_FOLDERS = []
        self.INCLUDE_SUB_FOLDERS = []

        # If set to False, the files will be *copied* to DST_ROOT_PATH, otherwise, the files will be *cut*/*moved* to DST_ROOT_PATH
        self.CUT_FILES = False
        
        # Raise exception when index couldn't be found in the config?
        self.RAISE_ON_MISSING_INDEX = True

        self.FILENAME_POSTFIX = ""


        ##################################
        
        self.CONFIG = {}
        self.CONFIG[self.KEY_MARKERS_ALIAS_ORDERED] = ["c1", "c2", "c3", "c4"]
            
class ConfigVer1(Config):
    def __init__(self):
        super().__init__()
        self.CONFIG[self.KEY_CELL_LINES] = {
                    "C9": {
                        "reps": {
                            "rep1": "C9-1",
                            "rep2": "C9-2",
                            "rep3": "C9 3"
                        }
                    },
                    "Ctrl": {
                        "reps": {
                            "rep1": "Ctrl 1",
                            "rep2": "Ctrl 2",
                            "rep3": "Ctrl 3"
                        }
                    },
                    "SALSPositive": {
                        "reps": {
                            "rep1": "SALS+ 1",
                            "rep2": "SALS+ 2",
                            "rep3": "SALS+ 3"
                        }
                    },
                    "SALSNegative": {
                        "reps": {
                            "rep1": "SALS-1",
                            "rep2": "SALS-2",
                            "rep3": "SALS-3"
                        }
                    }
                }

class ConfigVer2(Config):
    def __init__(self):
        super().__init__()
        self.CONFIG[self.KEY_CELL_LINES] = {
                "C9": {
                    "reps": {
                        "rep1": "C9 1",
                        "rep2": "C9 2",
                        "rep3": "C9 3"
                    }
                },
                "Ctrl": {
                    "reps": {
                        "rep1": "Ctrl 1",
                        "rep2": "Ctrl 2",
                        "rep3": "Ctrl 3"
                    }
                },
                "SALSPositive": {
                    "reps": {
                        "rep1": "SALS+ 1",
                        "rep2": "SALS+ 2",
                        "rep3": "SALS+ 3"
                    }
                },
                "SALSNegative": {
                    "reps": {
                        "rep1": "SALS-1",
                        "rep2": "SALS- 2",
                        "rep3": "SALS- 3"
                    }
                }
            }
        
class ConfigPanelA(ConfigVer1):
    def __init__(self):
        super().__init__()
        self.FOLDERS = ['panelA']
        self.CONFIG[self.KEY_MARKERS] = {
                "panelA": ["TDP43", "Map2", "DCP1A", "DAPI"]
            }

class ConfigPanelB(ConfigVer1):
    def __init__(self):
        super().__init__()
        self.FOLDERS = ['panelB']
        self.CONFIG[self.KEY_MARKERS] = {
                "panelB": ["LaminB1", "Nup62", "POM121", "DAPI"]
            }

class ConfigPanelC(ConfigVer1):
    def __init__(self):
        super().__init__()
        self.FOLDERS = ['panelC']
        self.CONFIG[self.KEY_MARKERS] = {
                "panelC": ["G3BP1", "PURA", "KIF5A", "DAPI"]
            }

class ConfigPanelD(ConfigVer1):
    def __init__(self):
        super().__init__()
        self.FOLDERS = ['panelD']
        self.CONFIG[self.KEY_MARKERS] = {
                "panelD": ["FUS", "CD41", "FMRP", "DAPI"]
            }

class ConfigPanelE(ConfigVer1):
    def __init__(self):
        super().__init__()
        self.FOLDERS = ['panelE']
        self.CONFIG[self.KEY_MARKERS] = {
                "panelE": ["TIA1", "Nup98", "TOMM20", "DAPI"]
            }

class ConfigPanelF(ConfigVer2):
    def __init__(self):
        super().__init__()
        self.FOLDERS = ['panelF']

        
        self.CONFIG[self.KEY_MARKERS] = {
                "panelF": ["SCNA", "Nup153", "ANXA11", "DAPI"]
            }

class ConfigPanelG(ConfigVer2):
    def __init__(self):
        super().__init__()
        self.FOLDERS = ['panelG']
        self.CONFIG[self.KEY_MARKERS] = {
                "panelG": ["SQSTM1", "PSD95", "Lamp1", "DAPI"]
            }

class ConfigPanelH(ConfigVer2):
    def __init__(self):
        super().__init__()
        self.FOLDERS = ['panelH']
        self.CONFIG[self.KEY_MARKERS] = {
                "panelH": ["NEMO", "Phalloidin", "NCL", "DAPI"]
            }


class ConfigPanelI(ConfigVer2):
    def __init__(self):
        super().__init__()
        self.FOLDERS = ['panelI']
        self.CONFIG[self.KEY_MARKERS_ALIAS_ORDERED] = ["c1", "c2", "c3"]

        self.CONFIG[self.KEY_MARKERS] = {
                "panelI": ["GM130", "Calreticulin", "DAPI"]
            }

class ConfigPanelJ(ConfigVer2):
    def __init__(self):
        super().__init__()
        self.FOLDERS = ['panelJ']
        self.CONFIG[self.KEY_MARKERS_ALIAS_ORDERED] = ["c1", "c2", "c3"]

        self.CONFIG[self.KEY_MARKERS] = {
                "panelJ": ["NONO", "CLTC", "DAPI"]
            }

class ConfigPanelK(ConfigVer2):
    def __init__(self):
        super().__init__()
        self.FOLDERS = ['panelK']
        self.CONFIG[self.KEY_MARKERS_ALIAS_ORDERED] = ["c1", "c2", "c3"]

        self.CONFIG[self.KEY_MARKERS] = {
                "panelK": ["hnRNPA1", "EEA1", "DAPI"]
            }

class ConfigPanelL(ConfigVer2):
    def __init__(self):
        super().__init__()
        self.FOLDERS = ['panelL']
        self.CONFIG[self.KEY_MARKERS_ALIAS_ORDERED] = ["c1", "c2", "c3"]

        self.CONFIG[self.KEY_MARKERS] = {
                "panelL": ["hnRNPA2B1", "Calnexin", "DAPI"]
            }