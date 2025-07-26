######################################################
########## Please Don't Change This Section ##########
######################################################

import os
import sys

sys.path.insert(1, os.getenv("NOVA_HOME"))
from tools.images_organizer.opera_version2.config import Config

class Config_A(Config):
    def __init__(self, batch_number:int):
        super().__init__()

        folder_name = f"Batch_{batch_number}"
        self.FOLDERS = [folder_name]
        # self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/images']
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/panelA']

        ##################################

        ########################################
        ############### Advanced ###############
        ########################################

        row1, row2 = 1,2
        dNLS_Untreated_cols = [1,2]
        dNLS_DOX_cols = [2,3]
        WT_Untreated_col = 4
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "dNLS": {
                    "DOX": [(row2,dNLS_DOX_cols[0]), (row1,dNLS_DOX_cols[1]), (row2,dNLS_DOX_cols[1])], 
                    "Untreated": [(row1,dNLS_Untreated_cols[0]), (row2,dNLS_Untreated_cols[0]), (row1,dNLS_Untreated_cols[1])]
                },
                "WT": {
                    "Untreated": [(row1,WT_Untreated_col), (row2,WT_Untreated_col)]
                },
            },
            # [DAPI, Cy3, Cy2, Cy5]
            # ch1 - DAPI
            # ch2 - Cy3 (mCherry)
            # ch3 - Cy2 (GFP)
            # ch4 - Cy5
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3", "ch4"],
            self.KEY_MARKERS: {
                'panelA': ["DAPI", "FMRP", "PURA", "G3BP1"]
                },
            self.KEY_REPS: ["rep1", "rep2", "rep3"],
        }

class Config_A_B1(Config_A):
    def __init__(self):
        super().__init__(batch_number=1)
class Config_A_B2(Config_A):
    def __init__(self):
        super().__init__(batch_number=2)
class Config_A_B3(Config_A):
    def __init__(self):
        super().__init__(batch_number=3)
class Config_A_B4(Config_A):
    def __init__(self):
        super().__init__(batch_number=4)
class Config_A_B5(Config_A):
    def __init__(self):
        super().__init__(batch_number=5)
class Config_A_B6(Config_A):
    def __init__(self):
        super().__init__(batch_number=6)

#######################################

class Config_B(Config):
    def __init__(self, batch_number:int):
        super().__init__()

        folder_name = f"Batch_{batch_number}"
        self.FOLDERS = [folder_name]
        # self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/images']
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/panelB']

        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        
        row1, row2 = 3,4
        dNLS_Untreated_cols = [1,2]
        dNLS_DOX_cols = [2,3]
        WT_Untreated_col = 4
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "dNLS": {
                    "DOX": [(row2,dNLS_DOX_cols[0]), (row1,dNLS_DOX_cols[1]), (row2,dNLS_DOX_cols[1])], 
                    "Untreated": [(row1,dNLS_Untreated_cols[0]), (row2,dNLS_Untreated_cols[0]), (row1,dNLS_Untreated_cols[1])]
                },
                "WT": {
                    "Untreated": [(row1,WT_Untreated_col), (row2,WT_Untreated_col)]
                },
            },
            # [DAPI, Cy3, Cy2, Cy5]
            # ch1 - DAPI
            # ch2 - Cy3 (mCherry)
            # ch3 - Cy2 (GFP)
            # ch4 - Cy5
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3", "ch4"],
            self.KEY_MARKERS: {
                'panelB': ["DAPI", "SON", "CD41", "NONO"]
                },
            self.KEY_REPS: ["rep1", "rep2", "rep3"],
        }

class Config_B_B1(Config_B):
    def __init__(self):
        super().__init__(batch_number=1)
class Config_B_B2(Config_B):
    def __init__(self):
        super().__init__(batch_number=2)
class Config_B_B3(Config_B):
    def __init__(self):
        super().__init__(batch_number=3)
class Config_B_B4(Config_B):
    def __init__(self):
        super().__init__(batch_number=4)
class Config_B_B5(Config_B):
    def __init__(self):
        super().__init__(batch_number=5)
class Config_B_B6(Config_B):
    def __init__(self):
        super().__init__(batch_number=6)

class Config_C(Config):
    def __init__(self, batch_number:int):
        super().__init__()

        folder_name = f"Batch_{batch_number}"
        self.FOLDERS = [folder_name]
        # self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/images']
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/panelC']

        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        
        row1, row2 = 5,6
        dNLS_Untreated_cols = [1,2]
        dNLS_DOX_cols = [2,3]
        WT_Untreated_col = 4
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "dNLS": {
                    "DOX": [(row2,dNLS_DOX_cols[0]), (row1,dNLS_DOX_cols[1]), (row2,dNLS_DOX_cols[1])], 
                    "Untreated": [(row1,dNLS_Untreated_cols[0]), (row2,dNLS_Untreated_cols[0]), (row1,dNLS_Untreated_cols[1])]
                },
                "WT": {
                    "Untreated": [(row1,WT_Untreated_col), (row2,WT_Untreated_col)]
                },
            },
            # [DAPI, Cy3, Cy2, Cy5]
            # ch1 - DAPI
            # ch2 - Cy3 (mCherry)
            # ch3 - Cy2 (GFP)
            # ch4 - Cy5
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3", "ch4"],
            self.KEY_MARKERS: {
                'panelC': ["DAPI", "KIF5A", "Tubulin", "SQSTM1"]
                },
            self.KEY_REPS: ["rep1", "rep2", "rep3"],
        }

class Config_C_B1(Config_C):
    def __init__(self):
        super().__init__(batch_number=1)
class Config_C_B2(Config_C):
    def __init__(self):
        super().__init__(batch_number=2)
class Config_C_B3(Config_C):
    def __init__(self):
        super().__init__(batch_number=3)
class Config_C_B4(Config_C):
    def __init__(self):
        super().__init__(batch_number=4)
class Config_C_B5(Config_C):
    def __init__(self):
        super().__init__(batch_number=5)
class Config_C_B6(Config_C):
    def __init__(self):
        super().__init__(batch_number=6)

class Config_D(Config): 
    def __init__(self, batch_number:int):
        super().__init__()

        folder_name = f"Batch_{batch_number}"
        self.FOLDERS = [folder_name]
        # self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/images']
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/panelD']

        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        
        row1, row2 = 7,8
        dNLS_Untreated_cols = [1,2]
        dNLS_DOX_cols = [2,3]
        WT_Untreated_col = 4
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "dNLS": {
                    "DOX": [(row2,dNLS_DOX_cols[0]), (row1,dNLS_DOX_cols[1]), (row2,dNLS_DOX_cols[1])], 
                    "Untreated": [(row1,dNLS_Untreated_cols[0]), (row2,dNLS_Untreated_cols[0]), (row1,dNLS_Untreated_cols[1])]
                },
                "WT": {
                    "Untreated": [(row1,WT_Untreated_col), (row2,WT_Untreated_col)]
                },
            },
            # [DAPI, Cy3, Cy2, Cy5]
            # ch1 - DAPI
            # ch2 - Cy3 (mCherry)
            # ch3 - Cy2 (GFP)
            # ch4 - Cy5
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3", "ch4"],
            self.KEY_MARKERS: {
                'panelD': ["DAPI", "CLTC", "Phalloidin", "PSD95"]
                },
            self.KEY_REPS: ["rep1", "rep2", "rep3"],
        }

class Config_D_B1(Config_D):
    def __init__(self):
        super().__init__(batch_number=1)
class Config_D_B2(Config_D):
    def __init__(self):
        super().__init__(batch_number=2)
class Config_D_B3(Config_D):
    def __init__(self):
        super().__init__(batch_number=3)
class Config_D_B4(Config_D):
    def __init__(self):
        super().__init__(batch_number=4)
class Config_D_B5(Config_D):
    def __init__(self):
        super().__init__(batch_number=5)
class Config_D_B6(Config_D):
    def __init__(self):
        super().__init__(batch_number=6)

class Config_E(Config):
    def __init__(self, batch_number:int):
        super().__init__()

        folder_name = f"Batch_{batch_number}"
        self.FOLDERS = [folder_name]
        # self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/images']
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/panelE']

        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        
        row1, row2 = 1,2
        dNLS_Untreated_cols = [5,6]
        dNLS_DOX_cols = [6,7]
        WT_Untreated_col = 8
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "dNLS": {
                    "DOX": [(row2,dNLS_DOX_cols[0]), (row1,dNLS_DOX_cols[1]), (row2,dNLS_DOX_cols[1])], 
                    "Untreated": [(row1,dNLS_Untreated_cols[0]), (row2,dNLS_Untreated_cols[0]), (row1,dNLS_Untreated_cols[1])]
                },
                "WT": {
                    "Untreated": [(row1,WT_Untreated_col), (row2,WT_Untreated_col)]
                },
            },
            # [DAPI, Cy5, Cy3]
            # ch1 - DAPI
            # ch2 - Cy5
            # ch3 - Cy3
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3"],
            self.KEY_MARKERS: {
                'panelE': ["DAPI", "NEMO", "DCP1A"]
                },
            self.KEY_REPS: ["rep1", "rep2", "rep3"],
        }

class Config_E_B1(Config_E):
    def __init__(self):
        super().__init__(batch_number=1)
class Config_E_B2(Config_E):
    def __init__(self):
        super().__init__(batch_number=2)
class Config_E_B3(Config_E):
    def __init__(self):
        super().__init__(batch_number=3)
class Config_E_B4(Config_E):
    def __init__(self):
        super().__init__(batch_number=4)
class Config_E_B5(Config_E):
    def __init__(self):
        super().__init__(batch_number=5)
class Config_E_B6(Config_E):
    def __init__(self):
        super().__init__(batch_number=6)

class Config_F(Config):
    def __init__(self, batch_number:int):
        super().__init__()

        folder_name = f"Batch_{batch_number}"
        self.FOLDERS = [folder_name]
        # self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/images']
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/panelF']

        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        
        row1, row2 = 3,4
        dNLS_Untreated_cols = [5,6]
        dNLS_DOX_cols = [6,7]
        WT_Untreated_col = 8
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "dNLS": {
                    "DOX": [(row2,dNLS_DOX_cols[0]), (row1,dNLS_DOX_cols[1]), (row2,dNLS_DOX_cols[1])], 
                    "Untreated": [(row1,dNLS_Untreated_cols[0]), (row2,dNLS_Untreated_cols[0]), (row1,dNLS_Untreated_cols[1])]
                },
                "WT": {
                    "Untreated": [(row1,WT_Untreated_col), (row2,WT_Untreated_col)]
                },
            },
            # [DAPI, Cy5, Cy3]
            # ch1 - DAPI
            # ch2 - Cy5
            # ch3 - Cy3
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3"],
            self.KEY_MARKERS: {
                'panelF': ["DAPI", "GM130", "Calreticulin"]
                },
            self.KEY_REPS: ["rep1", "rep2", "rep3"],
        }

class Config_F_B1(Config_F):
    def __init__(self):
        super().__init__(batch_number=1)
class Config_F_B2(Config_F):
    def __init__(self):
        super().__init__(batch_number=2)
class Config_F_B3(Config_F):
    def __init__(self):
        super().__init__(batch_number=3)
class Config_F_B4(Config_F):
    def __init__(self):
        super().__init__(batch_number=4)
class Config_F_B5(Config_F):
    def __init__(self):
        super().__init__(batch_number=5)
class Config_F_B6(Config_F):
    def __init__(self):
        super().__init__(batch_number=6)

class Config_G(Config):
    def __init__(self, batch_number:int):
        super().__init__()

        folder_name = f"Batch_{batch_number}"
        self.FOLDERS = [folder_name]
        # self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/images']
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/panelG']

        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        
        row1, row2 = 5,6
        dNLS_Untreated_cols = [5,6]
        dNLS_DOX_cols = [6,7]
        WT_Untreated_col = 8
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "dNLS": {
                    "DOX": [(row2,dNLS_DOX_cols[0]), (row1,dNLS_DOX_cols[1]), (row2,dNLS_DOX_cols[1])], 
                    "Untreated": [(row1,dNLS_Untreated_cols[0]), (row2,dNLS_Untreated_cols[0]), (row1,dNLS_Untreated_cols[1])]
                },
                "WT": {
                    "Untreated": [(row1,WT_Untreated_col), (row2,WT_Untreated_col)]
                },
            },
            # [DAPI, Cy5, Cy3]
            # ch1 - DAPI
            # ch2 - Cy5
            # ch3 - Cy3
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3"],
            self.KEY_MARKERS: {
                'panelG': ["DAPI", "NCL", "FUS"]
                },
            self.KEY_REPS: ["rep1", "rep2", "rep3"],
        }

class Config_G_B1(Config_G):
    def __init__(self):
        super().__init__(batch_number=1)
class Config_G_B2(Config_G):
    def __init__(self):
        super().__init__(batch_number=2)
class Config_G_B3(Config_G):
    def __init__(self):
        super().__init__(batch_number=3)
class Config_G_B4(Config_G):
    def __init__(self):
        super().__init__(batch_number=4)
class Config_G_B5(Config_G):
    def __init__(self):
        super().__init__(batch_number=5)
class Config_G_B6(Config_G):
    def __init__(self):
        super().__init__(batch_number=6)

class Config_H(Config):
    def __init__(self, batch_number:int):
        super().__init__()

        folder_name = f"Batch_{batch_number}"
        self.FOLDERS = [folder_name]
        # self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/images']
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/panelH']

        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        
        row1, row2 = 7,8
        dNLS_Untreated_cols = [5,6]
        dNLS_DOX_cols = [6,7]
        WT_Untreated_col = 8
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "dNLS": {
                    "DOX": [(row2,dNLS_DOX_cols[0]), (row1,dNLS_DOX_cols[1]), (row2,dNLS_DOX_cols[1])], 
                    "Untreated": [(row1,dNLS_Untreated_cols[0]), (row2,dNLS_Untreated_cols[0]), (row1,dNLS_Untreated_cols[1])]
                },
                "WT": {
                    "Untreated": [(row1,WT_Untreated_col), (row2,WT_Untreated_col)]
                },
            },
            # [DAPI, Cy5, Cy3]
            # ch1 - DAPI
            # ch2 - Cy5
            # ch3 - Cy3
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3"],
            self.KEY_MARKERS: {
                'panelH': ["DAPI", "LSM14A", "HNRNPA1"]
                },
            self.KEY_REPS: ["rep1", "rep2", "rep3"],
        }

class Config_H_B1(Config_H):
    def __init__(self):
        super().__init__(batch_number=1)
class Config_H_B2(Config_H):
    def __init__(self):
        super().__init__(batch_number=2)
class Config_H_B3(Config_H):
    def __init__(self):
        super().__init__(batch_number=3)
class Config_H_B4(Config_H):
    def __init__(self):
        super().__init__(batch_number=4)
class Config_H_B5(Config_H):
    def __init__(self):
        super().__init__(batch_number=5)
class Config_H_B6(Config_H):
    def __init__(self):
        super().__init__(batch_number=6)

class Config_I(Config):
    def __init__(self, batch_number:int):
        super().__init__()

        folder_name = f"Batch_{batch_number}"
        self.FOLDERS = [folder_name]
        # self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/images']
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/panelI']

        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        
        row1, row2 = 1,2
        dNLS_Untreated_cols = [9,10]
        dNLS_DOX_cols = [10,11]
        WT_Untreated_col = 12
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "dNLS": {
                    "DOX": [(row2,dNLS_DOX_cols[0]), (row1,dNLS_DOX_cols[1]), (row2,dNLS_DOX_cols[1])], 
                    "Untreated": [(row1,dNLS_Untreated_cols[0]), (row2,dNLS_Untreated_cols[0]), (row1,dNLS_Untreated_cols[1])]
                },
                "WT": {
                    "Untreated": [(row1,WT_Untreated_col), (row2,WT_Untreated_col)]
                },
            },
            # [DAPI, Cy3, Cy5]
            # ch1 - DAPI
            # ch2 - Cy3
            # ch3 - Cy5
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3"],
            self.KEY_MARKERS: {
                'panelI': ["DAPI", "PML", "TDP43"]
                },
            self.KEY_REPS: ["rep1", "rep2", "rep3"],
        }

class Config_I_B1(Config_I):
    def __init__(self):
        super().__init__(batch_number=1)
class Config_I_B2(Config_I):
    def __init__(self):
        super().__init__(batch_number=2)
class Config_I_B3(Config_I):
    def __init__(self):
        super().__init__(batch_number=3)
class Config_I_B4(Config_I):
    def __init__(self):
        super().__init__(batch_number=4)
class Config_I_B5(Config_I):
    def __init__(self):
        super().__init__(batch_number=5)
class Config_I_B6(Config_I):
    def __init__(self):
        super().__init__(batch_number=6)

class Config_J(Config):
    def __init__(self, batch_number:int):
        super().__init__()

        folder_name = f"Batch_{batch_number}"
        self.FOLDERS = [folder_name]
        # self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/images']
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/panelJ']

        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        
        row1, row2 = 3,4
        dNLS_Untreated_cols = [9,10]
        dNLS_DOX_cols = [10,11]
        WT_Untreated_col = 12
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "dNLS": {
                    "DOX": [(row2,dNLS_DOX_cols[0]), (row1,dNLS_DOX_cols[1]), (row2,dNLS_DOX_cols[1])], 
                    "Untreated": [(row1,dNLS_Untreated_cols[0]), (row2,dNLS_Untreated_cols[0]), (row1,dNLS_Untreated_cols[1])]
                },
                "WT": {
                    "Untreated": [(row1,WT_Untreated_col), (row2,WT_Untreated_col)]
                },
            },
            # [DAPI, Cy5, Cy3]
            # ch1 - DAPI
            # ch2 - Cy5
            # ch3 - Cy3
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3"],
            self.KEY_MARKERS: {
                'panelJ': ["DAPI", "ANXA11", "LAMP1"]
                },
            self.KEY_REPS: ["rep1", "rep2", "rep3"],
        }

class Config_J_B1(Config_J):
    def __init__(self):
        super().__init__(batch_number=1)
class Config_J_B2(Config_J):
    def __init__(self):
        super().__init__(batch_number=2)
class Config_J_B3(Config_J):
    def __init__(self):
        super().__init__(batch_number=3)
class Config_J_B4(Config_J):
    def __init__(self):
        super().__init__(batch_number=4)
class Config_J_B5(Config_J):
    def __init__(self):
        super().__init__(batch_number=5)
class Config_J_B6(Config_J):
    def __init__(self):
        super().__init__(batch_number=6)

class Config_K(Config):
    def __init__(self, batch_number:int):
        super().__init__()

        folder_name = f"Batch_{batch_number}"
        self.FOLDERS = [folder_name]
        # self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/images']
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/panelK']

        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        
        row1, row2 = 5,6
        dNLS_Untreated_cols = [9,10]
        dNLS_DOX_cols = [10,11]
        WT_Untreated_col = 12
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "dNLS": {
                    "DOX": [(row2,dNLS_DOX_cols[0]), (row1,dNLS_DOX_cols[1]), (row2,dNLS_DOX_cols[1])], 
                    "Untreated": [(row1,dNLS_Untreated_cols[0]), (row2,dNLS_Untreated_cols[0]), (row1,dNLS_Untreated_cols[1])]
                },
                "WT": {
                    "Untreated": [(row1,WT_Untreated_col), (row2,WT_Untreated_col)]
                },
            },
            # [DAPI, Cy5, Cy3]
            # ch1 - DAPI
            # ch2 - Cy5
            # ch3 - Cy3
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3"],
            self.KEY_MARKERS: {
                'panelK': ["DAPI", "PEX14", "SNCA"]
                },
            self.KEY_REPS: ["rep1", "rep2", "rep3"],
        }

class Config_K_B1(Config_K):
    def __init__(self):
        super().__init__(batch_number=1)
class Config_K_B2(Config_K):
    def __init__(self):
        super().__init__(batch_number=2)
class Config_K_B3(Config_K):
    def __init__(self):
        super().__init__(batch_number=3)
class Config_K_B4(Config_K):
    def __init__(self):
        super().__init__(batch_number=4)
class Config_K_B5(Config_K):
    def __init__(self):
        super().__init__(batch_number=5)
class Config_K_B6(Config_K):
    def __init__(self):
        super().__init__(batch_number=6)

class Config_L(Config):
    def __init__(self, batch_number:int):
        super().__init__()

        folder_name = f"Batch_{batch_number}"
        self.FOLDERS = [folder_name]
        # self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/images']
        self.INCLUDE_SUB_FOLDERS = [f'{folder_name}/panelL']

        ##################################

        ########################################
        ############### Advanced ###############
        ########################################
        
        row1, row2 = 7,8
        dNLS_Untreated_cols = [9,10]
        dNLS_DOX_cols = [10,11]
        WT_Untreated_col = 12
        
        self.CONFIG = {
            self.KEY_CELL_LINES: {
                "dNLS": {
                    "DOX": [(row2,dNLS_DOX_cols[0]), (row1,dNLS_DOX_cols[1]), (row2,dNLS_DOX_cols[1])], 
                    "Untreated": [(row1,dNLS_Untreated_cols[0]), (row2,dNLS_Untreated_cols[0]), (row1,dNLS_Untreated_cols[1])]
                },
                "WT": {
                    "Untreated": [(row1,WT_Untreated_col), (row2,WT_Untreated_col)]
                },
            },
           # [DAPI, Cy3, Cy2, Cy5]
            # ch1 - DAPI
            # ch2 - Cy3 (mCherry)
            # ch3 - Cy2 (GFP)
            # ch4 - Cy5
            self.KEY_MARKERS_ALIAS_ORDERED: ["ch1", "ch2", "ch3", "ch4"],
            self.KEY_MARKERS: {
                'panelL': ["DAPI", "TIA1", "TOMM20", "mitotracker"]
                },
            self.KEY_REPS: ["rep1", "rep2", "rep3"],
        }

class Config_L_B1(Config_L):
    def __init__(self):
        super().__init__(batch_number=1)
class Config_L_B2(Config_L):
    def __init__(self):
        super().__init__(batch_number=2)
class Config_L_B3(Config_L):
    def __init__(self):
        super().__init__(batch_number=3)
class Config_L_B4(Config_L):
    def __init__(self):
        super().__init__(batch_number=4)
class Config_L_B5(Config_L):
    def __init__(self):
        super().__init__(batch_number=5)
class Config_L_B6(Config_L):
    def __init__(self):
        super().__init__(batch_number=6)