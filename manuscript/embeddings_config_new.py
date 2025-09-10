import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.embeddings.embeddings_config import EmbeddingsConfig


################# Alyssa: EmbeddingsAlyssaCoyneDatasetConfig #######################

class EmbeddingsAlyssaCoyneDatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = None
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'AlyssaCoyne'    
        self.MARKERS_TO_EXCLUDE = ['MERGED']
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

        self.SHUFFLE:bool = False

        self.SETS:List[str] = ['testset']


class EmbeddingsAlyssaCoyneDatasetConfigCombined(EmbeddingsAlyssaCoyneDatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "AlyssaCoyne", f) for f in
                        ["batch1"]]
       

        self.MARKERS:List[str]            =  ["FUS", "TDP43"]

        # Cell lines to include
        self.CELL_LINES:List[str]         = ["c9orf72ALSPatients", "sALSNegativeCytoTDP43", "sALSPositiveCytoTDP43", "Controls"]

        # Conditions to include
        self.CONDITIONS:List[str]         = ["Untreated"]

## SUBSET CONFIGS:
class AlyssaCoyneC9vsControlSubset(EmbeddingsAlyssaCoyneDatasetConfigCombined):

    def __init__(self):
        super().__init__()

        # Cell lines to include
        self.CELL_LINES:List[str]         = ["c9orf72ALSPatients", "Controls"]

        # Conditions to include
        self.CONDITIONS:List[str]         = ["Untreated"]

class AlyssaCoyneNegativeTDP43vsControlSubset(EmbeddingsAlyssaCoyneDatasetConfigCombined):
    def __init__(self):
        super().__init__()

        self.CELL_LINES: List[str] = ["sALSNegativeCytoTDP43", "Controls"]
        self.CONDITIONS: List[str] = ["Untreated"]

class AlyssaCoynePositiveTDP43vsControlSubset(EmbeddingsAlyssaCoyneDatasetConfigCombined):
    def __init__(self):
        super().__init__()

        self.CELL_LINES: List[str] = ["sALSPositiveCytoTDP43", "Controls"]
        self.CONDITIONS: List[str] = ["Untreated"]

# ################# NEW dNLS: EmbeddingsNewdNLSCombinedDatasetConfig #######################

class EmbeddingsNewdNLSDatasetConfig(EmbeddingsConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = None
       
        self.SPLIT_DATA = False
        self.EXPERIMENT_TYPE = 'dNLS'
        self.MARKERS_TO_EXCLUDE = []
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class EmbeddingsNewdNLSDatasetConfigCombined(EmbeddingsNewdNLSDatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "dNLS", f) for f in
                        ["batch1", "batch2", "batch3", "batch4", "batch5", "batch6"]]

        self.SHUFFLE:bool = False

        self.SETS:List[str] = ['testset']

        self.MARKERS:List[str]            =  ["TDP43"]

        # Cell lines to include
        self.CELL_LINES:List[str]         = ["dNLS"]

        # Conditions to include
        self.CONDITIONS:List[str]         = ["DOX", "Untreated"]

# SUBSET 
class NewdNLSDoxVsUntreatedSubset(EmbeddingsNewdNLSDatasetConfigCombined):
    def __init__(self):
        super().__init__()

        self.CELL_LINES: List[str] = ["dNLS"]
        self.CONDITIONS: List[str] = ["DOX", "Untreated"]

# ################# NEW INDI: EmbeddingsDay8CombinedDatasetConfig #######################
# class EmbeddingsDay8NewDatasetConfig(EmbeddingsConfig):
#     def __init__(self):
#         super().__init__()

#         self.INPUT_FOLDERS = None
       
#         self.SPLIT_DATA = False
#         self.EXPERIMENT_TYPE = 'neuronsDay8_new'
#         self.MARKERS_TO_EXCLUDE = None
#         self.ADD_BATCH_TO_LABEL = True
#         self.ADD_REP_TO_LABEL = True



# class EmbeddingsDay8DatasetConfigCombined(EmbeddingsDay8NewDatasetConfig):
#     def __init__(self):
#         super().__init__()

#         self.SHUFFLE:bool = False

#         self.SETS:List[str] = ['testset']

#         # Conditions to include
#         self.CONDITIONS:List[str]         = ["Untreated"]

# class EmbeddingsDay8DatasetConfigBatch1(EmbeddingsDay8DatasetConfigCombined):
#     def __init__(self):
#         super().__init__()

#         self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "neuronsDay8_new", f) for f in
#                         ["batch1"]]

#         self.MARKERS:List[str]            =  [
#                                                 "FUS", "NEMO", "LSM14A", "SNCA", "Calreticulin", "DCP1A", "ANXA11",
#                                                 "HNRNPA1",  "LAMP1", "PEX14", "TOMM20", "TIA1", "Tubulin",
#                                                 "PSD95", "HNRNPA1", "CLTC", "NONO"
#                                             ]

#         # Cell lines to include
#         self.CELL_LINES:List[str]         = ["WT", "TDP43", "FUSHeterozygous", "FUSHomozygous", "OPTN", "TBK1"]



# # class NewIndiTBK1VsWTBatch1Subset(EmbeddingsDay8DatasetConfigBatch1):
# #     def __init__(self):
# #         super().__init__()
# #         self.CELL_LINES: List[str] = ["TBK1", "WT"]
# #         self.CONDITIONS: List[str] = ["Untreated"]
# #         self.MARKERS: List[str] = ["FUS", "NEMO", "LSM14A", "SNCA", "Calreticulin", "DCP1A", "ANXA11", "HNRNPA1"]

# # class NewIndiFUSHeteroVsWTBatch1Subset(EmbeddingsDay8DatasetConfigBatch1):
# #     def __init__(self):
# #         super().__init__()
# #         self.CELL_LINES: List[str] = ["FUSHeterozygous", "WT"]
# #         self.CONDITIONS: List[str] = ["Untreated"]
# #         self.MARKERS: List[str] = ["FUS", "LAMP1", "LSM14A", "PEX14", "Calreticulin", "HNRNPA1", "TOMM20"]

# # class NewIndiOPTNVsWTBatch1Subset(EmbeddingsDay8DatasetConfigBatch1):
# #     def __init__(self):
# #         super().__init__()
# #         self.CELL_LINES: List[str] = ["OPTN", "WT"]
# #         self.CONDITIONS: List[str] = ["Untreated"]
# #         self.MARKERS: List[str] =  ["CLTC"]#["FUS", "LAMP1", "TIA1", "NEMO", "Calreticulin", "Tubulin", "PSD95", "HNRNPA1", "CLTC"]

# class NewIndiFUSHomoVsWTBatch1Subset(EmbeddingsDay8DatasetConfigBatch1):
#     def __init__(self):
#         super().__init__()
#         self.CELL_LINES: List[str] = ["FUSHomozygous", "WT"]
#         self.CONDITIONS: List[str] = ["Untreated"]
#         self.MARKERS: List[str] = ["FUS", "LAMP1", "PEX14", "SNCA", "LSM14A", "HNRNPA1", "TIA1", "NEMO", "CLTC", "NONO"]

# class NewIndiTDP43vsWTBatch1Subset(EmbeddingsDay8DatasetConfigBatch1):
#     def __init__(self):
#         super().__init__()
#         self.CELL_LINES: List[str] = ["TDP43", "WT"]
#         self.CONDITIONS: List[str] = ["Untreated"]
#         self.MARKERS: List[str] = ["FUS", "LAMP1", "NEMO", "LSM14A", "Calreticulin", "CLTC"]



# class EmbeddingsDay8DatasetConfigBatch2(EmbeddingsDay8DatasetConfigCombined):
#     def __init__(self):
#         super().__init__()

#         self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "neuronsDay8_new", f) for f in
#                         ["batch2"]]

#         self.MARKERS:List[str]            =  [
#                                                 "FUS", "NEMO", "LSM14A", "SNCA", "Calreticulin", "DCP1A", "ANXA11",
#                                                 "HNRNPA1",  "LAMP1", "PEX14", "TOMM20", "TIA1", "Tubulin",
#                                                 "PSD95", "HNRNPA1", "CLTC", "NONO"
#                                             ]

#         # Cell lines to include
#         self.CELL_LINES:List[str]         = ["WT", "TDP43", "FUSHeterozygous", "FUSHomozygous", "OPTN", "TBK1"]



# # class NewIndiTBK1VsWTBatch2Subset(EmbeddingsDay8DatasetConfigBatch2):
# #     def __init__(self):
# #         super().__init__()
# #         self.CELL_LINES: List[str] = ["TBK1", "WT"]
# #         self.CONDITIONS: List[str] = ["Untreated"]
# #         self.MARKERS: List[str] = ["FUS", "NEMO", "LSM14A", "SNCA", "Calreticulin", "DCP1A", "ANXA11", "HNRNPA1"]

# # class NewIndiFUSHeteroVsWTBatch2Subset(EmbeddingsDay8DatasetConfigBatch2):
# #     def __init__(self):
# #         super().__init__()
# #         self.CELL_LINES: List[str] = ["FUSHeterozygous", "WT"]
# #         self.CONDITIONS: List[str] = ["Untreated"]
# #         self.MARKERS: List[str] = ["FUS", "LAMP1", "LSM14A", "PEX14", "Calreticulin", "HNRNPA1", "TOMM20"]

# # class NewIndiOPTNVsWTBatch2Subset(EmbeddingsDay8DatasetConfigBatch2):
# #     def __init__(self):
# #         super().__init__()
# #         self.CELL_LINES: List[str] = ["OPTN", "WT"]
# #         self.CONDITIONS: List[str] = ["Untreated"]
# #         self.MARKERS: List[str] =  ["CLTC"]#["FUS", "LAMP1", "TIA1", "NEMO", "Calreticulin", "Tubulin", "PSD95", "HNRNPA1", "CLTC"]

# class NewIndiFUSHomoVsWTBatch2Subset(EmbeddingsDay8DatasetConfigBatch2):
#     def __init__(self):
#         super().__init__()
#         self.CELL_LINES: List[str] = ["FUSHomozygous", "WT"]
#         self.CONDITIONS: List[str] = ["Untreated"]
#         self.MARKERS: List[str] = ["FUS", "LAMP1", "PEX14", "SNCA", "LSM14A", "HNRNPA1", "TIA1", "NEMO", "CLTC", "NONO"]

# class NewIndiTDP43vsWTBatch2Subset(EmbeddingsDay8DatasetConfigBatch2):
#     def __init__(self):
#         super().__init__()
#         self.CELL_LINES: List[str] = ["TDP43", "WT"]
#         self.CONDITIONS: List[str] = ["Untreated"]
#         self.MARKERS: List[str] = ["FUS", "LAMP1", "NEMO", "LSM14A", "Calreticulin", "CLTC"]

# class EmbeddingsDay8DatasetConfigBatch3(EmbeddingsDay8DatasetConfigCombined):
#     def __init__(self):
#         super().__init__()

#         self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "neuronsDay8_new", f) for f in
#                         ["batch3"]]

#         self.MARKERS:List[str]            =  [
#                                                 "FUS", "NEMO", "LSM14A", "SNCA", "Calreticulin", "DCP1A", "ANXA11",
#                                                 "HNRNPA1",  "LAMP1", "PEX14", "TOMM20", "TIA1", "Tubulin",
#                                                 "PSD95", "HNRNPA1", "CLTC", "NONO"
#                                             ]

#         # Cell lines to include
#         self.CELL_LINES:List[str]         = ["WT", "TDP43", "FUSHeterozygous", "FUSHomozygous", "OPTN", "TBK1"]



# # class NewIndiTBK1VsWTBatch3Subset(EmbeddingsDay8DatasetConfigBatch3):
# #     def __init__(self):
# #         super().__init__()
# #         self.CELL_LINES: List[str] = ["TBK1", "WT"]
# #         self.CONDITIONS: List[str] = ["Untreated"]
# #         self.MARKERS: List[str] = ["FUS", "NEMO", "LSM14A", "SNCA", "Calreticulin", "DCP1A", "ANXA11", "HNRNPA1"]

# # class NewIndiFUSHeteroVsWTBatch3Subset(EmbeddingsDay8DatasetConfigBatch3):
# #     def __init__(self):
# #         super().__init__()
# #         self.CELL_LINES: List[str] = ["FUSHeterozygous", "WT"]
# #         self.CONDITIONS: List[str] = ["Untreated"]
# #         self.MARKERS: List[str] = ["FUS", "LAMP1", "LSM14A", "PEX14", "Calreticulin", "HNRNPA1", "TOMM20"]

# # class NewIndiOPTNVsWTBatch3Subset(EmbeddingsDay8DatasetConfigBatch3):
# #     def __init__(self):
# #         super().__init__()
# #         self.CELL_LINES: List[str] = ["OPTN", "WT"]
# #         self.CONDITIONS: List[str] = ["Untreated"]
# #         self.MARKERS: List[str] =  ["CLTC"]#["FUS", "LAMP1", "TIA1", "NEMO", "Calreticulin", "Tubulin", "PSD95", "HNRNPA1", "CLTC"]

# class NewIndiFUSHomoVsWTBatch3Subset(EmbeddingsDay8DatasetConfigBatch3):
#     def __init__(self):
#         super().__init__()
#         self.CELL_LINES: List[str] = ["FUSHomozygous", "WT"]
#         self.CONDITIONS: List[str] = ["Untreated"]
#         self.MARKERS: List[str] = ["FUS", "LAMP1", "PEX14", "SNCA", "LSM14A", "HNRNPA1", "TIA1", "NEMO", "CLTC", "NONO"]

# class NewIndiTDP43vsWTBatch3Subset(EmbeddingsDay8DatasetConfigBatch3):
#     def __init__(self):
#         super().__init__()
#         self.CELL_LINES: List[str] = ["TDP43", "WT"]
#         self.CONDITIONS: List[str] = ["Untreated"]
#         self.MARKERS: List[str] = ["FUS", "LAMP1", "NEMO", "LSM14A", "Calreticulin", "CLTC"]

# class EmbeddingsDay8DatasetConfigBatch8(EmbeddingsDay8DatasetConfigCombined):
#     def __init__(self):
#         super().__init__()

#         self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "neuronsDay8_new", f) for f in
#                         ["batch8"]]

#         self.MARKERS:List[str]            =  [
#                                                 "FUS", "NEMO", "LSM14A", "SNCA", "Calreticulin", "DCP1A", "ANXA11",
#                                                 "HNRNPA1",  "LAMP1", "PEX14", "TOMM20", "TIA1", "Tubulin",
#                                                 "PSD95", "HNRNPA1", "CLTC", "NONO"
#                                             ]

#         # Cell lines to include
#         self.CELL_LINES:List[str]         = ["WT", "TDP43", "FUSHeterozygous", "FUSHomozygous", "OPTN", "TBK1"]



# # class NewIndiTBK1VsWTBatch8Subset(EmbeddingsDay8DatasetConfigBatch8):
# #     def __init__(self):
# #         super().__init__()
# #         self.CELL_LINES: List[str] = ["TBK1", "WT"]
# #         self.CONDITIONS: List[str] = ["Untreated"]
# #         self.MARKERS: List[str] = ["FUS", "NEMO", "LSM14A", "SNCA", "Calreticulin", "DCP1A", "ANXA11", "HNRNPA1"]

# # class NewIndiFUSHeteroVsWTBatch8Subset(EmbeddingsDay8DatasetConfigBatch8):
# #     def __init__(self):
# #         super().__init__()
# #         self.CELL_LINES: List[str] = ["FUSHeterozygous", "WT"]
# #         self.CONDITIONS: List[str] = ["Untreated"]
# #         self.MARKERS: List[str] = ["FUS", "LAMP1", "LSM14A", "PEX14", "Calreticulin", "HNRNPA1", "TOMM20"]

# # class NewIndiOPTNVsWTBatch8Subset(EmbeddingsDay8DatasetConfigBatch8):
# #     def __init__(self):
# #         super().__init__()
# #         self.CELL_LINES: List[str] = ["OPTN", "WT"]
# #         self.CONDITIONS: List[str] = ["Untreated"]
# #         self.MARKERS: List[str] =  ["CLTC"]#["FUS", "LAMP1", "TIA1", "NEMO", "Calreticulin", "Tubulin", "PSD95", "HNRNPA1", "CLTC"]

# class NewIndiFUSHomoVsWTBatch8Subset(EmbeddingsDay8DatasetConfigBatch8):
#     def __init__(self):
#         super().__init__()
#         self.CELL_LINES: List[str] = ["FUSHomozygous", "WT"]
#         self.CONDITIONS: List[str] = ["Untreated"]
#         self.MARKERS: List[str] = ["FUS", "LAMP1", "PEX14", "SNCA", "LSM14A", "HNRNPA1", "TIA1", "NEMO", "CLTC", "NONO"]

# class NewIndiTDP43vsWTBatch8Subset(EmbeddingsDay8DatasetConfigBatch8):
#     def __init__(self):
#         super().__init__()
#         self.CELL_LINES: List[str] = ["TDP43", "WT"]
#         self.CONDITIONS: List[str] = ["Untreated"]
#         self.MARKERS: List[str] = ["FUS", "LAMP1", "NEMO", "LSM14A", "Calreticulin", "CLTC"]

# class EmbeddingsDay8DatasetConfigBatch9(EmbeddingsDay8DatasetConfigCombined):
#     def __init__(self):
#         super().__init__()

#         self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "neuronsDay8_new", f) for f in
#                         ["batch9"]]

#         self.MARKERS:List[str]            =  [
#                                                 "FUS", "NEMO", "LSM14A", "SNCA", "Calreticulin", "DCP1A", "ANXA11",
#                                                 "HNRNPA1",  "LAMP1", "PEX14", "TOMM20", "TIA1", "Tubulin",
#                                                 "PSD95", "HNRNPA1", "CLTC", "NONO"
#                                             ]

#         # Cell lines to include
#         self.CELL_LINES:List[str]         = ["WT", "TDP43", "FUSHeterozygous", "FUSHomozygous", "OPTN", "TBK1"]



# # class NewIndiTBK1VsWTBatch9Subset(EmbeddingsDay8DatasetConfigBatch9):
# #     def __init__(self):
# #         super().__init__()
# #         self.CELL_LINES: List[str] = ["TBK1", "WT"]
# #         self.CONDITIONS: List[str] = ["Untreated"]
# #         self.MARKERS: List[str] = ["FUS", "NEMO", "LSM14A", "SNCA", "Calreticulin", "DCP1A", "ANXA11", "HNRNPA1"]

# # class NewIndiFUSHeteroVsWTBatch9Subset(EmbeddingsDay8DatasetConfigBatch9):
# #     def __init__(self):
# #         super().__init__()
# #         self.CELL_LINES: List[str] = ["FUSHeterozygous", "WT"]
# #         self.CONDITIONS: List[str] = ["Untreated"]
# #         self.MARKERS: List[str] = ["FUS", "LAMP1", "LSM14A", "PEX14", "Calreticulin", "HNRNPA1", "TOMM20"]

# # class NewIndiOPTNVsWTBatch9Subset(EmbeddingsDay8DatasetConfigBatch9):
# #     def __init__(self):
# #         super().__init__()
# #         self.CELL_LINES: List[str] = ["OPTN", "WT"]
# #         self.CONDITIONS: List[str] = ["Untreated"]
# #         self.MARKERS: List[str] =  ["CLTC"]#["FUS", "LAMP1", "TIA1", "NEMO", "Calreticulin", "Tubulin", "PSD95", "HNRNPA1", "CLTC"]

# class NewIndiFUSHomoVsWTBatch9Subset(EmbeddingsDay8DatasetConfigBatch9):
#     def __init__(self):
#         super().__init__()
#         self.CELL_LINES: List[str] = ["FUSHomozygous", "WT"]
#         self.CONDITIONS: List[str] = ["Untreated"]
#         self.MARKERS: List[str] = ["FUS", "LAMP1", "PEX14", "SNCA", "LSM14A", "HNRNPA1", "TIA1", "NEMO", "CLTC", "NONO"]

# class NewIndiTDP43vsWTBatch9Subset(EmbeddingsDay8DatasetConfigBatch9):
#     def __init__(self):
#         super().__init__()
#         self.CELL_LINES: List[str] = ["TDP43", "WT"]
#         self.CONDITIONS: List[str] = ["Untreated"]
#         self.MARKERS: List[str] = ["FUS", "LAMP1", "NEMO", "LSM14A", "Calreticulin", "CLTC"]

# class EmbeddingsDay8DatasetConfigBatch10(EmbeddingsDay8DatasetConfigCombined):
#     def __init__(self):
#         super().__init__()

#         self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "ManuscriptFinalData_80pct", "neuronsDay8_new", f) for f in
#                         ["batch10"]]

#         self.MARKERS:List[str]            =  [
#                                                 "FUS", "NEMO", "LSM14A", "SNCA", "Calreticulin", "DCP1A", "ANXA11",
#                                                 "HNRNPA1",  "LAMP1", "PEX14", "TOMM20", "TIA1", "Tubulin",
#                                                 "PSD95", "HNRNPA1", "CLTC", "NONO"
#                                             ]

#         # Cell lines to include
#         self.CELL_LINES:List[str]         = ["WT", "TDP43", "FUSHeterozygous", "FUSHomozygous", "OPTN", "TBK1"]



# # class NewIndiTBK1VsWTBatch10Subset(EmbeddingsDay8DatasetConfigBatch10):
# #     def __init__(self):
# #         super().__init__()
# #         self.CELL_LINES: List[str] = ["TBK1", "WT"]
# #         self.CONDITIONS: List[str] = ["Untreated"]
# #         self.MARKERS: List[str] = ["FUS", "NEMO", "LSM14A", "SNCA", "Calreticulin", "DCP1A", "ANXA11", "HNRNPA1"]

# # class NewIndiFUSHeteroVsWTBatch10Subset(EmbeddingsDay8DatasetConfigBatch10):
# #     def __init__(self):
# #         super().__init__()
# #         self.CELL_LINES: List[str] = ["FUSHeterozygous", "WT"]
# #         self.CONDITIONS: List[str] = ["Untreated"]
# #         self.MARKERS: List[str] = ["FUS", "LAMP1", "LSM14A", "PEX14", "Calreticulin", "HNRNPA1", "TOMM20"]

# # class NewIndiOPTNVsWTBatch10Subset(EmbeddingsDay8DatasetConfigBatch10):
# #     def __init__(self):
# #         super().__init__()
# #         self.CELL_LINES: List[str] = ["OPTN", "WT"]
# #         self.CONDITIONS: List[str] = ["Untreated"]
# #         self.MARKERS: List[str] =  ["CLTC"]#["FUS", "LAMP1", "TIA1", "NEMO", "Calreticulin", "Tubulin", "PSD95", "HNRNPA1", "CLTC"]

# class NewIndiFUSHomoVsWTBatch10Subset(EmbeddingsDay8DatasetConfigBatch10):
#     def __init__(self):
#         super().__init__()
#         self.CELL_LINES: List[str] = ["FUSHomozygous", "WT"]
#         self.CONDITIONS: List[str] = ["Untreated"]
#         self.MARKERS: List[str] = ["FUS", "LAMP1", "PEX14", "SNCA", "LSM14A", "HNRNPA1", "TIA1", "NEMO", "CLTC", "NONO"]

# class NewIndiTDP43vsWTBatch10Subset(EmbeddingsDay8DatasetConfigBatch10):
#     def __init__(self):
#         super().__init__()
#         self.CELL_LINES: List[str] = ["TDP43", "WT"]
#         self.CONDITIONS: List[str] = ["Untreated"]
#         self.MARKERS: List[str] = ["FUS", "LAMP1", "NEMO", "LSM14A", "Calreticulin", "CLTC"]



