######################################################
########## Please Don't Change This Section ##########
######################################################

import os

class Config():
    def __init__(self):
        super().__init__()
        
        self.LOGGING_PATH = os.path.join('tools', 'images_organizer', 'AlyssaCoyne', 'logs', 'Coyne150725') #os.path.join('tools', 'images_organizer', 'opera_version2', 'logs', 'iAstrocytes')
        self.KEY_CELL_LINES = "cell_lines"
        self.KEY_MARKERS_ALIAS_ORDERED = "markers_alias_ordered"
        self.KEY_MARKERS = "markers"
        self.KEY_BATCHES = "batches"
        self.KEY_REPS = "reps"
        self.FILE_EXTENSION = ".tif"
        self.KEY_COL_WELLS = "wells_columns"
        self.KEY_ROW_WELLS = "wells_rows"
        self.KEY_REP_PREFIX = "rep"
        self.KEY_ALIAS = "alias"

        #####################################
        ############### Paths ###############
        #####################################

        # Path to source folder (root)
        self.SRC_ROOT_PATH = "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/raw/AlyssaCoyne_new"
        # Path to destination folder (root)
        self.DST_ROOT_PATH = "/home/projects/hornsteinlab/Collaboration/NOVA/input/images/raw/AlyssaCoyne_new_sorted/batch1"

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
        # c1: Alexa 647
        # c2: Alexa 568
        # c3: Alexa 488
        # c4: DAPI
            
class ConfigVer1(Config):
    def __init__(self):
        super().__init__()
        self.CONFIG[self.KEY_CELL_LINES] = {
                    "c9 CS2YNL": {
                        self.KEY_ALIAS: 'C9-CS2YNL',
                        self.KEY_REPS: {
                            f'{self.KEY_REP_PREFIX}1': (31,35),
                            f'{self.KEY_REP_PREFIX}2': (36,40)
                        }
                    },
                    "c9 CS7VCZ": {
                        self.KEY_ALIAS: 'C9-CS7VCZ',
                        self.KEY_REPS: {
                            f'{self.KEY_REP_PREFIX}1': (41,45),
                            f'{self.KEY_REP_PREFIX}2': (46,50)
                        }
                    },
                    "c9 CS8RFT": {
                        self.KEY_ALIAS: 'C9-CS8RFT',
                        self.KEY_REPS: {
                            f'{self.KEY_REP_PREFIX}1': (51,55),
                            f'{self.KEY_REP_PREFIX}2': (56,60)
                        }
                    },
                    
                    "control EDi022": {
                        self.KEY_ALIAS: 'Ctrl-EDi022',
                        self.KEY_REPS: {
                            f'{self.KEY_REP_PREFIX}1': (11,15),
                            f'{self.KEY_REP_PREFIX}2': (16,20)
                        }
                    },
                    "control EDi029": {
                        self.KEY_ALIAS: 'Ctrl-EDi029',
                        self.KEY_REPS: {
                            f'{self.KEY_REP_PREFIX}1': (21,25),
                            f'{self.KEY_REP_PREFIX}2': (26,30)
                        }
                    },
                    "control EDi037": {
                        self.KEY_ALIAS: 'Ctrl-EDi037',
                        self.KEY_REPS: {
                            f'{self.KEY_REP_PREFIX}1': (1,5),
                            f'{self.KEY_REP_PREFIX}2': (6,10)
                        }
                    },

                    "sALS+ CS2FN3": {
                        self.KEY_ALIAS: 'SALSPositive-CS2FN3',
                        self.KEY_REPS: {
                            f'{self.KEY_REP_PREFIX}1': (91,95),
                            f'{self.KEY_REP_PREFIX}2': (96,100)
                        }
                    },
                    "sALS+ CS4ZCD": {
                        self.KEY_ALIAS: 'SALSPositive-CS4ZCD',
                        self.KEY_REPS: {
                            f'{self.KEY_REP_PREFIX}1': (101,105),
                            f'{self.KEY_REP_PREFIX}2': (106,110)
                        }
                    },
                    "sALS+ CS7TN6": {
                        self.KEY_ALIAS: 'SALSPositive-CS7TN6',
                        self.KEY_REPS: {
                            f'{self.KEY_REP_PREFIX}1': (111,115),
                            f'{self.KEY_REP_PREFIX}2': (116,120)
                        }
                    },

                    "sALS- CS0ANK": {
                        self.KEY_ALIAS: 'SALSNegative-CS0ANK',
                        self.KEY_REPS: {
                            f'{self.KEY_REP_PREFIX}1': (61,65),
                            f'{self.KEY_REP_PREFIX}2': (66,70)
                        }
                    },
                    "sALS- CS0JPP": {
                        self.KEY_ALIAS: 'SALSNegative-CS0JPP',
                        self.KEY_REPS: {
                            f'{self.KEY_REP_PREFIX}1': (71,75),
                            f'{self.KEY_REP_PREFIX}2': (76,80)
                        }
                    },
                    "sALS- CS6ZU8": {
                        self.KEY_ALIAS: 'SALSNegative-CS6ZU8',
                        self.KEY_REPS: {
                            f'{self.KEY_REP_PREFIX}1': (81,85),
                            f'{self.KEY_REP_PREFIX}2': (86,90)
                        }
                    }
                }

class ConfigVer2(Config):
    def __init__(self):
        super().__init__()
        self.CONFIG[self.KEY_CELL_LINES] = {
            "c9 CS2YNL": {
                self.KEY_ALIAS: 'C9-CS2YNL',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (31,35),
                    f'{self.KEY_REP_PREFIX}2': (36,40)
                }
            },
            "c9 CS7VCZ": {
                self.KEY_ALIAS: 'C9-CS7VCZ',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (41,45),
                    f'{self.KEY_REP_PREFIX}2': (46,50)
                }
            },
            "c9 CS8RFT": {
                self.KEY_ALIAS: 'C9-CS8RFT',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (51,55),
                    f'{self.KEY_REP_PREFIX}2': (56,60)
                }
            },
            
            "control EDi022": {
                self.KEY_ALIAS: 'Ctrl-EDi022',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (1,5),
                    f'{self.KEY_REP_PREFIX}2': (6,10)
                }
            },
            "control EDi029": {
                self.KEY_ALIAS: 'Ctrl-EDi029',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (11,15),
                    f'{self.KEY_REP_PREFIX}2': (16,20)
                }
            },
            "control EDi037": {
                self.KEY_ALIAS: 'Ctrl-EDi037',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': 	(21,25),
                    f'{self.KEY_REP_PREFIX}2': (26,30)
                }
            },

            "sALS+ CS2FN3": {
                self.KEY_ALIAS: 'SALSPositive-CS2FN3',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': 	(111, 115),
                    f'{self.KEY_REP_PREFIX}2': (116, 120)
                }
            },
           "sALS+ CS4ZCD": {
                self.KEY_ALIAS: 'SALSPositive-CS4ZCD',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': 	(61,65),
                    f'{self.KEY_REP_PREFIX}2': (66,70)
                }
            },
            "sALS+ CS7TN6": {
                self.KEY_ALIAS: 'SALSPositive-CS7TN6',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': 	(71,75),
                    f'{self.KEY_REP_PREFIX}2': 	(76,80)
                }
            },

            "sALS- CS0ANK": {
                self.KEY_ALIAS: 'SALSNegative-CS0ANK',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (81,85),
                    f'{self.KEY_REP_PREFIX}2': (86,90)
                }
            },
            "sALS- CS0JPP": {
                self.KEY_ALIAS: 'SALSNegative-CS0JPP',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (91,95),
                    f'{self.KEY_REP_PREFIX}2': (96, 100)
                }
            },
            "sALS- CS6ZU8": {
                self.KEY_ALIAS: 'SALSNegative-CS6ZU8',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (101, 105),
                    f'{self.KEY_REP_PREFIX}2': (106, 110)
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
        # Here TDP43 nd DCP1A were switched, hence:
        # c1: Alexa 488
        # c2: Alexa 568
        # c3: Alexa 647
        # c4: DAPI

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
    # NOTE: Extra tile in c9 CS2YNL
    def __init__(self):
        super().__init__()
        self.FOLDERS = ['panelI']
        self.CONFIG[self.KEY_MARKERS_ALIAS_ORDERED] = ["c1", "c2", "c3"]

        self.CONFIG[self.KEY_MARKERS] = {
                "panelI": ["GM130", "Calreticulin", "DAPI"]
            }

        self.CONFIG[self.KEY_CELL_LINES] = {
            "c9 CS2YNL": {
                self.KEY_ALIAS: 'C9-CS2YNL',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (31,35),
                    f'{self.KEY_REP_PREFIX}2': (36,41)
                }
            },
            "c9 CS7VCZ": {
                self.KEY_ALIAS: 'C9-CS7VCZ',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (42,46),
                    f'{self.KEY_REP_PREFIX}2': (47,51)
                }
            },
            "c9 CS8RFT": {
                self.KEY_ALIAS: 'C9-CS8RFT',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (52,56),
                    f'{self.KEY_REP_PREFIX}2': (57,61)
                }
            },
            
            "control EDi022": {
                self.KEY_ALIAS: 'Ctrl-EDi022',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (1,5),
                    f'{self.KEY_REP_PREFIX}2': (6,10)
                }
            },
            "control EDi029": {
                self.KEY_ALIAS: 'Ctrl-EDi029',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (11,15),
                    f'{self.KEY_REP_PREFIX}2': (16,20)
                }
            },
            "control EDi037": {
                self.KEY_ALIAS: 'Ctrl-EDi037',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (21,25),
                    f'{self.KEY_REP_PREFIX}2': (26,30)
                }
            },

            "sALS+ CS2FN3": {
                self.KEY_ALIAS: 'SALSPositive-CS2FN3',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (112,116),
                    f'{self.KEY_REP_PREFIX}2': (117,121)
                }
            },
            "sALS+ CS4ZCD": {
                self.KEY_ALIAS: 'SALSPositive-CS4ZCD',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (62,66),
                    f'{self.KEY_REP_PREFIX}2': (67,71)
                }
            },
            "sALS+ CS7TN6": {
                self.KEY_ALIAS: 'SALSPositive-CS7TN6',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (72,76),
                    f'{self.KEY_REP_PREFIX}2': (77,81)
                }
            },

            "sALS- CS0ANK": {
                self.KEY_ALIAS: 'SALSNegative-CS0ANK',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (82,86),
                    f'{self.KEY_REP_PREFIX}2': (87,91)
                }
            },
            "sALS- CS0JPP": {
                self.KEY_ALIAS: 'SALSNegative-CS0JPP',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (92,96),
                    f'{self.KEY_REP_PREFIX}2': (97,101)
                }
            },
            "sALS- CS6ZU8": {
                self.KEY_ALIAS: 'SALSNegative-CS6ZU8',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (102,106),
                    f'{self.KEY_REP_PREFIX}2': (107,111)
                }
            }
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
    # NOTE: The same as I
    def __init__(self):
        super().__init__()
        self.FOLDERS = ['panelK']
        self.CONFIG[self.KEY_MARKERS_ALIAS_ORDERED] = ["c1", "c2", "c3"]

        self.CONFIG[self.KEY_MARKERS] = {
                "panelK": ["hnRNPA1", "EEA1", "DAPI"]
            }
        
        self.CONFIG[self.KEY_CELL_LINES] = {
            "c9 CS2YNL": {
                self.KEY_ALIAS: 'C9-CS2YNL',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (31,35),
                    f'{self.KEY_REP_PREFIX}2': (36,41)
                }
            },
            "c9 CS7VCZ": {
                self.KEY_ALIAS: 'C9-CS7VCZ',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (42,46),
                    f'{self.KEY_REP_PREFIX}2': (47,51)
                }
            },
            "c9 CS8RFT": {
                self.KEY_ALIAS: 'C9-CS8RFT',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (52,56),
                    f'{self.KEY_REP_PREFIX}2': (57,61)
                }
            },
            
            "control EDi022": {
                self.KEY_ALIAS: 'Ctrl-EDi022',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (1,5),
                    f'{self.KEY_REP_PREFIX}2': (6,10)
                }
            },
            "control EDi029": {
                self.KEY_ALIAS: 'Ctrl-EDi029',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (11,15),
                    f'{self.KEY_REP_PREFIX}2': (16,20)
                }
            },
            "control EDi037": {
                self.KEY_ALIAS: 'Ctrl-EDi037',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (21,25),
                    f'{self.KEY_REP_PREFIX}2': (26,30)
                }
            },

            "sALS+ CS2FN3": {
                self.KEY_ALIAS: 'SALSPositive-CS2FN3',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (112,116),
                    f'{self.KEY_REP_PREFIX}2': (117,121)
                }
            },
            "sALS+ CS4ZCD": {
                self.KEY_ALIAS: 'SALSPositive-CS4ZCD',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (62,66),
                    f'{self.KEY_REP_PREFIX}2': (67,71)
                }
            },
            "sALS+ CS7TN6": {
                self.KEY_ALIAS: 'SALSPositive-CS7TN6',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (72,76),
                    f'{self.KEY_REP_PREFIX}2': (77,81)
                }
            },

            "sALS- CS0ANK": {
                self.KEY_ALIAS: 'SALSNegative-CS0ANK',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (82,86),
                    f'{self.KEY_REP_PREFIX}2': (87,91)
                }
            },
            "sALS- CS0JPP": {
                self.KEY_ALIAS: 'SALSNegative-CS0JPP',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (92,96),
                    f'{self.KEY_REP_PREFIX}2': (97,101)
                }
            },
            "sALS- CS6ZU8": {
                self.KEY_ALIAS: 'SALSNegative-CS6ZU8',
                self.KEY_REPS: {
                    f'{self.KEY_REP_PREFIX}1': (102,106),
                    f'{self.KEY_REP_PREFIX}2': (107,111)
                }
            }
        }


class ConfigPanelL(ConfigVer2):
    def __init__(self):
        super().__init__()
        self.FOLDERS = ['panelL']
        self.CONFIG[self.KEY_MARKERS_ALIAS_ORDERED] = ["c1", "c2", "c3"]

        self.CONFIG[self.KEY_MARKERS] = {
                "panelL": ["hnRNPA2B1", "Calnexin", "DAPI"]
            }