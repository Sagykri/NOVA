######################################################
########## Please Don't Change This Section ##########
######################################################

import os

class Config():
    def __init__(self):
        super().__init__()
        
        self.CELL_LINES = ['Controls',
                           'sALS_Negative_cytoTDP43',
                           'sALS_Positive_cytoTDP43',
                           'c9orf72_ALS_patients']
        
        self.MARKERS = ['DAPI',
                        'TDP43',
                        'Map2',
                        'DCP1A'
        ]
        
        self.MARKER_ANTIGEN_NAMES_MAPPER = {
                                            '1': {'marker_name': 'DCP1A', 'antigen_name': 'confCy5'},
                                            '2': {'marker_name': 'Map2', 'antigen_name': 'confmCherry'},
                                            '3': {'marker_name': 'TDP43', 'antigen_name': 'confGFP'},
                                            '4': {'marker_name': 'DAPI', 'antigen_name': 'confDAPI'},
                                            '1-4': {'marker_name': 'MERGED', 'antigen_name': 'MERGED'}
        }
            
            
        self.FILE_EXTENSION = ".tif"
        

        #####################################
        ############### Paths ###############
        #####################################

        # Path to source folder (root)
        self.SRC_ROOT_PATH = "/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/AlyssaCoyne/unsorted_MOmaps_iPSC_patients_TDP43_PB_CoyneLab"
        # Path to destination folder (root)
        self.DST_ROOT_PATH = "/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/AlyssaCoyne/MOmaps_iPSC_patients_TDP43_PB_CoyneLab"

        # Names of folders to handle
        self.FOLDERS = ['Controls',
                        'sALS_Negative_cytoTDP43',
                        'sALS_Positive_cytoTDP43',
                        'c9orf72_ALS_patients']


        self.EXCLUDE_SUB_FOLDERS = []
        self.INCLUDE_SUB_FOLDERS = []

        # If set to False, the files will be *copied* to DST_ROOT_PATH, otherwise, the files will be *cut*/*moved* to DST_ROOT_PATH
        self.CUT_FILES = False
        
        # Raise exception when index couldn't be found in the config?
        self.RAISE_ON_MISSING_INDEX = True

        ##################################
        
