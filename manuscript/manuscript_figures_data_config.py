import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from src.figures.figures_config import FigureConfig
from manuscript.plot_config import PlotConfig
from src.datasets.label_utils import MapLabelsFunction


################
## New Alyssa
################

# UMAP0
class newAlyssaFigureConfig_UMAP0_B1(FigureConfig):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch1']
        
        self.EXPERIMENT_TYPE = 'AlyssaCoyne_new'    
        self.CELL_LINES = None
        self.CONDITIONS = ['Untreated']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False #True
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False

class newAlyssaFigureConfig_UMAP0_B1_per_gene_group(newAlyssaFigureConfig_UMAP0_B1):
    def __init__(self):
        super().__init__()
        
        self.REMOVE_PATIENT_ID_FROM_CELL_LINE = True

# UMAP 1

class newAlyssaFigureConfig_UMAP1_B1(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch1']
        
        self.EXPERIMENT_TYPE = 'AlyssaCoyne_new'    
        self.CELL_LINES = None
        self.CONDITIONS = ['Untreated']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False #True
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False

class newAlyssaFigureConfig_UMAP1_B1_C9_CS2YNL(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['C9-CS2YNL']

class newAlyssaFigureConfig_UMAP1_B1_C9_CS7VCZ(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['C9-CS7VCZ']

class newAlyssaFigureConfig_UMAP1_B1_C9_CS8RFT(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['C9-CS8RFT']

class newAlyssaFigureConfig_UMAP1_B1_Ctrl_EDi022(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['Ctrl-EDi022']

class newAlyssaFigureConfig_UMAP1_B1_Ctrl_EDi029(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['Ctrl-EDi029']

class newAlyssaFigureConfig_UMAP1_B1_Ctrl_EDi037(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['Ctrl-EDi037']

class newAlyssaFigureConfig_UMAP1_B1_SALSNegative_CS0ANK(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['SALSNegative-CS0ANK']

class newAlyssaFigureConfig_UMAP1_B1_SALSNegative_CS0JPP(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['SALSNegative-CS0JPP']

class newAlyssaFigureConfig_UMAP1_B1_SALSNegative_CS6ZU8(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['SALSNegative-CS6ZU8']

class newAlyssaFigureConfig_UMAP1_B1_SALSPositive_CS2FN3(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['SALSPositive-CS2FN3']

class newAlyssaFigureConfig_UMAP1_B1_SALSPositive_CS4ZCD(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['SALSPositive-CS4ZCD']

class newAlyssaFigureConfig_UMAP1_B1_SALSPositive_CS7TN6(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['SALSPositive-CS7TN6']


class newAlyssaFigureConfig_UMAP1_B1_C9(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['C9-CS2YNL', 'C9-CS7VCZ', 'C9-CS8RFT']

class newAlyssaFigureConfig_UMAP1_B1_Ctrl(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['Ctrl-EDi022', 'Ctrl-EDi029', 'Ctrl-EDi037']

class newAlyssaFigureConfig_UMAP1_B1_SALSNegative(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['SALSNegative-CS0ANK', 'SALSNegative-CS0JPP', 'SALSNegative-CS6ZU8']

class newAlyssaFigureConfig_UMAP1_B1_SALSPositive(newAlyssaFigureConfig_UMAP1_B1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """
        self.CELL_LINES = ['SALSPositive-CS2FN3', 'SALSPositive-CS4ZCD', 'SALSPositive-CS7TN6']


# UMAP2
class newAlyssaFigureConfig_UMAP2_B1(FigureConfig):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch1']
        
        self.EXPERIMENT_TYPE = 'AlyssaCoyne_new'    
        self.CONDITIONS = ['Untreated']
        self.CELL_LINES = None
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False#True
        self.ADD_REP_TO_LABEL=False
        self.ADD_BATCH_TO_LABEL = False

        self.REMOVE_PATIENT_ID_FROM_CELL_LINE = True

class newAlyssaFigureConfig_UMAP2_B1_with_patientID(newAlyssaFigureConfig_UMAP2_B1):
    def __init__(self):
        super().__init__()

        self.REMOVE_PATIENT_ID_FROM_CELL_LINE = False

class newAlyssaFigureConfig_UMAP2_B1_P1(FigureConfig):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch1']
        

        self.EXPERIMENT_TYPE = 'AlyssaCoyne_new'    
        self.CONDITIONS = ['Untreated']
        self.CELL_LINES = ['Ctrl-EDi022', 'C9-CS2YNL', 'SALSPositive-CS2FN3', 'SALSNegative-CS0ANK']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False#True
        self.ADD_REP_TO_LABEL=False
        self.ADD_BATCH_TO_LABEL = False

        self.REMOVE_PATIENT_ID_FROM_CELL_LINE = False

class newAlyssaFigureConfig_UMAP2_B1_P2(FigureConfig):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch1']
        

        self.EXPERIMENT_TYPE = 'AlyssaCoyne_new'    
        self.CONDITIONS = ['Untreated']
        self.CELL_LINES = ['Ctrl-EDi029', 'C9-CS7VCZ', 'SALSPositive-CS4ZCD', 'SALSNegative-CS0JPP']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False#True
        self.ADD_REP_TO_LABEL=False
        self.ADD_BATCH_TO_LABEL = False

        self.REMOVE_PATIENT_ID_FROM_CELL_LINE = False

class newAlyssaFigureConfig_UMAP2_B1_P3(FigureConfig):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch1']
        

        self.EXPERIMENT_TYPE = 'AlyssaCoyne_new'    
        self.CONDITIONS = ['Untreated']
        self.CELL_LINES = ['Ctrl-EDi037', 'C9-CS8RFT', 'SALSPositive-CS7TN6', 'SALSNegative-CS6ZU8']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False#True
        self.ADD_REP_TO_LABEL=False
        self.ADD_BATCH_TO_LABEL = False

        self.REMOVE_PATIENT_ID_FROM_CELL_LINE = False


class newAlyssaFigureConfig_UMAP2_4Markers_with_patientID(newAlyssaFigureConfig_UMAP2_B1_with_patientID):
    def __init__(self):
        super().__init__()

        self.MARKERS = ['DAPI', 'DCP1A', 'TDP43', 'Map2']

# Effect size
class AlyssaCoyneEffectsFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch1"]
        
        self.EXPERIMENT_TYPE = 'AlyssaCoyne'
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True

class AlyssaCoyneNEWEffectsFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch1"]
        
        self.EXPERIMENT_TYPE = 'AlyssaCoyne_new'
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.BASELINE_PERTURB = {'Ctrl-EDi022_Untreated':
                                 ['C9-CS2YNL_Untreated','SALSPositive-CS2FN3_Untreated','SALSNegative-CS0ANK_Untreated'],
                                 'Ctrl-EDi029_Untreated':
                                 ['C9-CS7VCZ_Untreated','SALSPositive-CS4ZCD_Untreated','SALSNegative-CS0JPP_Untreated'],
                                 'Ctrl-EDi037_Untreated':
                                 ['C9-CS8RFT_Untreated','SALSPositive-CS7TN6_Untreated','SALSNegative-CS6ZU8_Untreated']}



#############
## New iNDI
############

# UMAP1

class newNeuronsD8FigureConfig_UMAP1(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3', 'batch7', 'batch8', 'batch9']
        
        self.EXPERIMENT_TYPE = 'neuronsDay8_new'    
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False #True
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False

class newNeuronsD8FigureConfig_UMAP1_NIHMarkers(newNeuronsD8FigureConfig_UMAP1):
    def __init__(self):
        super().__init__()

        self.MARKERS_TO_EXCLUDE = ['LSM14A', 'SON', 'HNRNPA1']
        


class newNeuronsD8FigureConfig_UMAP1_B1(newNeuronsD8FigureConfig_UMAP1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        self.INPUT_FOLDERS = ['batch1']

class newNeuronsD8FigureConfig_UMAP1_B2(newNeuronsD8FigureConfig_UMAP1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        self.INPUT_FOLDERS = ['batch2']

class newNeuronsD8FigureConfig_UMAP1_B3(newNeuronsD8FigureConfig_UMAP1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        self.INPUT_FOLDERS = ['batch3']

class newNeuronsD8FigureConfig_UMAP1_B7(newNeuronsD8FigureConfig_UMAP1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        self.INPUT_FOLDERS = ['batch7']

class newNeuronsD8FigureConfig_UMAP1_B8(newNeuronsD8FigureConfig_UMAP1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        self.INPUT_FOLDERS = ['batch8']

class newNeuronsD8FigureConfig_UMAP1_B9(newNeuronsD8FigureConfig_UMAP1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        self.INPUT_FOLDERS = ['batch9']

class newNeuronsD8FigureConfig_UMAP1_B10(newNeuronsD8FigureConfig_UMAP1):
    def __init__(self):
        super().__init__()
        
        """UMAP1 of WT untreated
        """

        self.INPUT_FOLDERS = ['batch10']

# UMAP0

class newNeuronsD8FigureConfig_UMAP0(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3', 'batch7', 'batch8', 'batch9', 'batch10']
        
        self.EXPERIMENT_TYPE = 'neuronsDay8_new' #'neuronsDay8_new'    
        # self.CELL_LINES = ['WT']
        
        self.MARKERS_TO_EXCLUDE = ['DAPI', 'CD41']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False #True
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False

class newNeuronsD8FigureConfig_UMAP0_B2_WT(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch2']
        
        self.EXPERIMENT_TYPE = 'neuronsDay8_new' #'neuronsDay8_new'    
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        
        # self.MARKERS_TO_EXCLUDE = ['DAPI']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False #True
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False

class newNeuronsD8FigureConfig_UMAP0_B2_TDP43(newNeuronsD8FigureConfig_UMAP0_B2_WT):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        self.CELL_LINES = ['TDP43']

class newNeuronsD8FigureConfig_UMAP0_B2_TBK1(newNeuronsD8FigureConfig_UMAP0_B2_WT):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        self.CELL_LINES = ['TBK1']

class newNeuronsD8FigureConfig_UMAP0_B2_OPTN(newNeuronsD8FigureConfig_UMAP0_B2_WT):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        self.CELL_LINES = ['OPTN']

class newNeuronsD8FigureConfig_UMAP0_B2_FUSRevertant(newNeuronsD8FigureConfig_UMAP0_B2_WT):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        self.CELL_LINES = ['FUSRevertant']

class newNeuronsD8FigureConfig_UMAP0_B2_FUSHomozygous(newNeuronsD8FigureConfig_UMAP0_B2_WT):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        self.CELL_LINES = ['FUSHomozygous']

class newNeuronsD8FigureConfig_UMAP0_B2_FUSHeterozygous(newNeuronsD8FigureConfig_UMAP0_B2_WT):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

        self.CELL_LINES = ['FUSHeterozygous']

##

class newNeuronsD8FigureConfig_UMAP0_B1_WT_FUSREV_FUSMarker(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch1']
        
        self.EXPERIMENT_TYPE = 'neuronsDay8_new' #'neuronsDay8_new'    
        self.CELL_LINES = ['WT', 'FUSRevertant']
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['FUS']
        
        # self.MARKERS_TO_EXCLUDE = ['DAPI']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False #True
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False

class newNeuronsD8FigureConfig_UMAP0_B3_WT_FUSREV(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch3']
        
        self.EXPERIMENT_TYPE = 'neuronsDay8_new' #'neuronsDay8_new'    
        self.CELL_LINES = ['WT', 'FUSRevertant']
        self.CONDITIONS = ['Untreated']
        
        # self.MARKERS_TO_EXCLUDE = ['DAPI']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False #True
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False


class newNeuronsD8FigureConfig_UMAP0_B1(newNeuronsD8FigureConfig_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch1']

class newNeuronsD8FigureConfig_UMAP0_B2(newNeuronsD8FigureConfig_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch2']

class newNeuronsD8FigureConfig_UMAP0_B3(newNeuronsD8FigureConfig_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch3']

class newNeuronsD8FigureConfig_UMAP0_B7(newNeuronsD8FigureConfig_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch7']

class newNeuronsD8FigureConfig_UMAP0_B8(newNeuronsD8FigureConfig_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch8']

class newNeuronsD8FigureConfig_UMAP0_B9(newNeuronsD8FigureConfig_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch9']

class newNeuronsD8FigureConfig_UMAP0_B10(newNeuronsD8FigureConfig_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch10']


# UMAP0 ALS
class newNeuronsD8FigureConfig_ALS_UMAP0(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3', 'batch7', 'batch8', 'batch9', 'batch10']
        
        self.EXPERIMENT_TYPE = 'neuronsDay8_new' # 'neuronsDay8_new'    
        self.CONDITIONS = ['Untreated']
        self.CELL_LINES = None
        self.MARKERS_TO_EXCLUDE = ['DAPI', 'CD41']
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False #True
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False



#### UMAP 0 all batches all lines
class newNeuronsD8FigureConfig_ALS_UMAP0_allBatches_allALSLines(newNeuronsD8FigureConfig_ALS_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """
        self.CELL_LINES = ['FUSHeterozygous','FUSHomozygous','FUSRevertant','OPTN','TDP43','TBK1', 'WT']

### UMAP0 per batch

class newNeuronsD8FigureConfig_ALS_UMAP0_B1(newNeuronsD8FigureConfig_ALS_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch1']

class newNeuronsD8FigureConfig_ALS_UMAP0_B2(newNeuronsD8FigureConfig_ALS_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch2']

class newNeuronsD8FigureConfig_ALS_UMAP0_B3(newNeuronsD8FigureConfig_ALS_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch3']

class newNeuronsD8FigureConfig_ALS_UMAP0_B7(newNeuronsD8FigureConfig_ALS_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch7']

class newNeuronsD8FigureConfig_ALS_UMAP0_B8(newNeuronsD8FigureConfig_ALS_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch8']

class newNeuronsD8FigureConfig_ALS_UMAP0_B9(newNeuronsD8FigureConfig_ALS_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch9']

class newNeuronsD8FigureConfig_ALS_UMAP0_B10(newNeuronsD8FigureConfig_ALS_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch10']

# UMAP0 - WT and ALS line

# B1-10
class newNeuronsD8FigureConfig_UMAP0_allBatches_WT_TDP43(newNeuronsD8FigureConfig_ALS_UMAP0):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'TDP43']

class newNeuronsD8FigureConfig_UMAP0_allBatches_WT_OPTN(newNeuronsD8FigureConfig_ALS_UMAP0):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'OPTN']

class newNeuronsD8FigureConfig_UMAP0_allBatches_WT_TBK1(newNeuronsD8FigureConfig_ALS_UMAP0):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'TBK1']


class newNeuronsD8FigureConfig_UMAP0_allBatches_WT_FUSRev(newNeuronsD8FigureConfig_ALS_UMAP0):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'FUSRevertant']

class newNeuronsD8FigureConfig_UMAP0_allBatches_WT_FUSHet(newNeuronsD8FigureConfig_ALS_UMAP0):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'FUSHeterozygous']

class newNeuronsD8FigureConfig_UMAP0_allBatches_WT_FUSHom(newNeuronsD8FigureConfig_ALS_UMAP0):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'FUSHomozygous']


#B1

class newNeuronsD8FigureConfig_UMAP0_B1_WT_TDP43(newNeuronsD8FigureConfig_ALS_UMAP0_B1):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'TDP43']

class newNeuronsD8FigureConfig_UMAP0_B1_WT_OPTN(newNeuronsD8FigureConfig_ALS_UMAP0_B1):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'OPTN']

class newNeuronsD8FigureConfig_UMAP0_B1_WT_TBK1(newNeuronsD8FigureConfig_ALS_UMAP0_B1):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'TBK1']


class newNeuronsD8FigureConfig_UMAP0_B1_WT_FUSRev(newNeuronsD8FigureConfig_ALS_UMAP0_B1):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'FUSRevertant']

class newNeuronsD8FigureConfig_UMAP0_B1_WT_FUSHet(newNeuronsD8FigureConfig_ALS_UMAP0_B1):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'FUSHeterozygous']

class newNeuronsD8FigureConfig_UMAP0_B1_WT_FUSHom(newNeuronsD8FigureConfig_ALS_UMAP0_B1):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'FUSHomozygous']

#B2
class newNeuronsD8FigureConfig_UMAP0_B2_WT_TDP43(newNeuronsD8FigureConfig_ALS_UMAP0_B2):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'TDP43']

class newNeuronsD8FigureConfig_UMAP0_B2_WT_OPTN(newNeuronsD8FigureConfig_ALS_UMAP0_B2):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'OPTN']

class newNeuronsD8FigureConfig_UMAP0_B2_WT_TBK1(newNeuronsD8FigureConfig_ALS_UMAP0_B2):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'TBK1']


class newNeuronsD8FigureConfig_UMAP0_B2_WT_FUSRev(newNeuronsD8FigureConfig_ALS_UMAP0_B2):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'FUSRevertant']

class newNeuronsD8FigureConfig_UMAP0_B2_WT_FUSHet(newNeuronsD8FigureConfig_ALS_UMAP0_B2):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'FUSHeterozygous']

class newNeuronsD8FigureConfig_UMAP0_B2_WT_FUSHom(newNeuronsD8FigureConfig_ALS_UMAP0_B2):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'FUSHomozygous']

# B3

class newNeuronsD8FigureConfig_UMAP0_B3_WT_TDP43(newNeuronsD8FigureConfig_ALS_UMAP0_B3):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'TDP43']

class newNeuronsD8FigureConfig_UMAP0_B3_WT_OPTN(newNeuronsD8FigureConfig_ALS_UMAP0_B3):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'OPTN']

class newNeuronsD8FigureConfig_UMAP0_B3_WT_TBK1(newNeuronsD8FigureConfig_ALS_UMAP0_B3):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'TBK1']


class newNeuronsD8FigureConfig_UMAP0_B3_WT_FUSRev(newNeuronsD8FigureConfig_ALS_UMAP0_B3):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'FUSRevertant']

class newNeuronsD8FigureConfig_UMAP0_B3_WT_FUSHet(newNeuronsD8FigureConfig_ALS_UMAP0_B3):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'FUSHeterozygous']

class newNeuronsD8FigureConfig_UMAP0_B3_WT_FUSHom(newNeuronsD8FigureConfig_ALS_UMAP0_B3):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'FUSHomozygous']

# B7
class newNeuronsD8FigureConfig_UMAP0_B7_WT_TDP43(newNeuronsD8FigureConfig_ALS_UMAP0_B7):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'TDP43']

class newNeuronsD8FigureConfig_UMAP0_B7_WT_OPTN(newNeuronsD8FigureConfig_ALS_UMAP0_B7):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'OPTN']

class newNeuronsD8FigureConfig_UMAP0_B7_WT_TBK1(newNeuronsD8FigureConfig_ALS_UMAP0_B7):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'TBK1']


class newNeuronsD8FigureConfig_UMAP0_B7_WT_FUSRev(newNeuronsD8FigureConfig_ALS_UMAP0_B7):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'FUSRevertant']

class newNeuronsD8FigureConfig_UMAP0_B7_WT_FUSHet(newNeuronsD8FigureConfig_ALS_UMAP0_B7):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'FUSHeterozygous']

class newNeuronsD8FigureConfig_UMAP0_B7_WT_FUSHom(newNeuronsD8FigureConfig_ALS_UMAP0_B7):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'FUSHomozygous']

# B8
class newNeuronsD8FigureConfig_UMAP0_B8_WT_TDP43(newNeuronsD8FigureConfig_ALS_UMAP0_B8):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'TDP43']

class newNeuronsD8FigureConfig_UMAP0_B8_WT_OPTN(newNeuronsD8FigureConfig_ALS_UMAP0_B8):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'OPTN']

class newNeuronsD8FigureConfig_UMAP0_B8_WT_TBK1(newNeuronsD8FigureConfig_ALS_UMAP0_B8):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'TBK1']


class newNeuronsD8FigureConfig_UMAP0_B8_WT_FUSRev(newNeuronsD8FigureConfig_ALS_UMAP0_B8):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'FUSRevertant']

class newNeuronsD8FigureConfig_UMAP0_B8_WT_FUSHet(newNeuronsD8FigureConfig_ALS_UMAP0_B8):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'FUSHeterozygous']

class newNeuronsD8FigureConfig_UMAP0_B8_WT_FUSHom(newNeuronsD8FigureConfig_ALS_UMAP0_B8):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'FUSHomozygous']

# B9

class newNeuronsD8FigureConfig_UMAP0_B9_WT_TDP43(newNeuronsD8FigureConfig_ALS_UMAP0_B9):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'TDP43']

class newNeuronsD8FigureConfig_UMAP0_B9_WT_OPTN(newNeuronsD8FigureConfig_ALS_UMAP0_B9):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'OPTN']

class newNeuronsD8FigureConfig_UMAP0_B9_WT_TBK1(newNeuronsD8FigureConfig_ALS_UMAP0_B9):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'TBK1']


class newNeuronsD8FigureConfig_UMAP0_B9_WT_FUSRev(newNeuronsD8FigureConfig_ALS_UMAP0_B9):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'FUSRevertant']

class newNeuronsD8FigureConfig_UMAP0_B9_WT_FUSHet(newNeuronsD8FigureConfig_ALS_UMAP0_B9):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'FUSHeterozygous']

class newNeuronsD8FigureConfig_UMAP0_B9_WT_FUSHom(newNeuronsD8FigureConfig_ALS_UMAP0_B9):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'FUSHomozygous']

# B10
class newNeuronsD8FigureConfig_UMAP0_B10_WT_TDP43(newNeuronsD8FigureConfig_ALS_UMAP0_B10):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'TDP43']

class newNeuronsD8FigureConfig_UMAP0_B10_WT_OPTN(newNeuronsD8FigureConfig_ALS_UMAP0_B10):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'OPTN']

class newNeuronsD8FigureConfig_UMAP0_B10_WT_TBK1(newNeuronsD8FigureConfig_ALS_UMAP0_B10):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'TBK1']


class newNeuronsD8FigureConfig_UMAP0_B10_WT_FUSRev(newNeuronsD8FigureConfig_ALS_UMAP0_B10):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'FUSRevertant']

class newNeuronsD8FigureConfig_UMAP0_B10_WT_FUSHet(newNeuronsD8FigureConfig_ALS_UMAP0_B10):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'FUSHeterozygous']

class newNeuronsD8FigureConfig_UMAP0_B10_WT_FUSHom(newNeuronsD8FigureConfig_ALS_UMAP0_B10):
    def __init__(self):
        super().__init__()
        self.CELL_LINES = ['WT', 'FUSHomozygous']



# UMAP0 FUS lines
class newNeuronsD8FigureConfig_FUSLines_UMAP0(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3', 'batch7', 'batch8', 'batch9', 'batch10']
        
        self.EXPERIMENT_TYPE = 'neuronsDay8_new' #'neuronsDay8_new'    
        self.CELL_LINES = ['WT', 'FUSHomozygous','FUSHeterozygous','FUSRevertant']
        self.CONDITIONS = ['Untreated']
        self.MARKERS = None
        self.MARKERS_TO_EXCLUDE = ['DAPI']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False #True
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = False

class newNeuronsD8FigureConfig_FUSLines_UMAP0_B1(newNeuronsD8FigureConfig_FUSLines_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch1']

class newNeuronsD8FigureConfig_FUSLines_UMAP0_B2(newNeuronsD8FigureConfig_FUSLines_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch2']

class newNeuronsD8FigureConfig_FUSLines_UMAP0_B3(newNeuronsD8FigureConfig_FUSLines_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch3']

class newNeuronsD8FigureConfig_FUSLines_UMAP0_B7(newNeuronsD8FigureConfig_FUSLines_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch7']

class newNeuronsD8FigureConfig_FUSLines_UMAP0_B8(newNeuronsD8FigureConfig_FUSLines_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch8']

class newNeuronsD8FigureConfig_FUSLines_UMAP0_B9(newNeuronsD8FigureConfig_FUSLines_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch9']

class newNeuronsD8FigureConfig_FUSLines_UMAP0_B10(newNeuronsD8FigureConfig_FUSLines_UMAP0):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch10']

class newNeuronsD8FigureConfig_WT_vs_FUSLines_FUSMarker(FigureConfig):
    def __init__(self):
        super().__init__()
        
        """UMAP0 of WT untreated
        """

         
        self.INPUT_FOLDERS = ['batch1']
        
        self.EXPERIMENT_TYPE = 'neuronsDay8_new'    
        self.CELL_LINES = ['WT', 'FUSRevertant']
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['FUS']
        
        self.ADD_REP_TO_LABEL = False
        self.ADD_BATCH_TO_LABEL = True

# UMAP2
class newNeuronsD8FigureConfig_UMAP2(FigureConfig):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3', 'batch7', 'batch8', 'batch9']

        self.EXPERIMENT_TYPE = 'neuronsDay8_new'
        self.CONDITIONS = ['Untreated']
        
        self.CELL_LINES = ['WT', 'TDP43', 'OPTN', 'TBK1', 'FUSRevertant', 'FUSHeterozygous', 'FUSHomozygous']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False#True
        self.ADD_REP_TO_LABEL=False
        self.ADD_BATCH_TO_LABEL = False

class newNeuronsD8FigureConfig_UMAP2_B1(newNeuronsD8FigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch1']

class newNeuronsD8FigureConfig_UMAP2_B1_wo_FUSMarker(newNeuronsD8FigureConfig_UMAP2_B1):
    def __init__(self):
        super().__init__()

         
        self.MARKERS_TO_EXCLUDE = self.MARKERS_TO_EXCLUDE + ['FUS']

class newNeuronsD8FigureConfig_UMAP2_B2(newNeuronsD8FigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch2']

class newNeuronsD8FigureConfig_UMAP2_B2_wo_FUSMarker(newNeuronsD8FigureConfig_UMAP2_B2):
    def __init__(self):
        super().__init__()

         
        self.MARKERS_TO_EXCLUDE = self.MARKERS_TO_EXCLUDE + ['FUS']


class newNeuronsD8FigureConfig_UMAP2_B3(newNeuronsD8FigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch3']

class newNeuronsD8FigureConfig_UMAP2_B3_wo_FUSMarker(newNeuronsD8FigureConfig_UMAP2_B3):
    def __init__(self):
        super().__init__()

         
        self.MARKERS_TO_EXCLUDE = self.MARKERS_TO_EXCLUDE + ['FUS']

class newNeuronsD8FigureConfig_UMAP2_B7(newNeuronsD8FigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch7']

class newNeuronsD8FigureConfig_UMAP2_B7_wo_FUSMarker(newNeuronsD8FigureConfig_UMAP2_B7):
    def __init__(self):
        super().__init__()

         
        self.MARKERS_TO_EXCLUDE = self.MARKERS_TO_EXCLUDE + ['FUS']

class newNeuronsD8FigureConfig_UMAP2_B8(newNeuronsD8FigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch8']

class newNeuronsD8FigureConfig_UMAP2_B8_wo_FUSMarker(newNeuronsD8FigureConfig_UMAP2_B8):
    def __init__(self):
        super().__init__()

        self.MARKERS_TO_EXCLUDE = self.MARKERS_TO_EXCLUDE + ['FUS']

class newNeuronsD8FigureConfig_UMAP2_B9(newNeuronsD8FigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ['batch9']

class newNeuronsD8FigureConfig_UMAP2_B9_wo_FUSMarker(newNeuronsD8FigureConfig_UMAP2_B9):
    def __init__(self):
        super().__init__()

        self.MARKERS_TO_EXCLUDE = self.MARKERS_TO_EXCLUDE + ['FUS']

class newNeuronsD8FigureConfig_UMAP2_B10(newNeuronsD8FigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ['batch10']

class newNeuronsD8FigureConfig_UMAP2_B10_wo_FUSMarker(newNeuronsD8FigureConfig_UMAP2_B10):
    def __init__(self):
        super().__init__()

        self.MARKERS_TO_EXCLUDE = self.MARKERS_TO_EXCLUDE + ['FUS']

## WO FUSRev and wo FUSHet
class newNeuronsD8FigureConfig_UMAP2_WO_FUSRev_WO_FUSHet(FigureConfig):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3', 'batch7', 'batch8', 'batch9']

        self.EXPERIMENT_TYPE = 'neuronsDay8_new'
        self.CONDITIONS = ['Untreated']
        
        self.CELL_LINES = ['WT', 'TDP43', 'OPTN', 'TBK1', 'FUSHomozygous']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False#True
        self.ADD_REP_TO_LABEL=False
        self.ADD_BATCH_TO_LABEL = False

class newNeuronsD8FigureConfig_UMAP2_WO_FUSRev_WO_FUSHet_B1(newNeuronsD8FigureConfig_UMAP2_WO_FUSRev_WO_FUSHet):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch1']

class newNeuronsD8FigureConfig_UMAP2_WO_FUSRev_WO_FUSHet_B2(newNeuronsD8FigureConfig_UMAP2_WO_FUSRev_WO_FUSHet):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch2']


class newNeuronsD8FigureConfig_UMAP2_WO_FUSRev_WO_FUSHet_B3(newNeuronsD8FigureConfig_UMAP2_WO_FUSRev_WO_FUSHet):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch3']

class newNeuronsD8FigureConfig_UMAP2_WO_FUSRev_WO_FUSHet_B7(newNeuronsD8FigureConfig_UMAP2_WO_FUSRev_WO_FUSHet):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch7']

class newNeuronsD8FigureConfig_UMAP2_WO_FUSRev_WO_FUSHet_B8(newNeuronsD8FigureConfig_UMAP2_WO_FUSRev_WO_FUSHet):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch8']

class newNeuronsD8FigureConfig_UMAP2_WO_FUSRev_WO_FUSHet_B9(newNeuronsD8FigureConfig_UMAP2_WO_FUSRev_WO_FUSHet):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ['batch9']

# Without FUSRev and FUSHet, with SNCA
class newNeuronsD8FigureConfig_UMAP2_WO_FUSRev_WO_FUSHet_WithSNCA(newNeuronsD8FigureConfig_UMAP2_WO_FUSRev_WO_FUSHet):
    def __init__(self):
        super().__init__()
 
        self.INPUT_FOLDERS = ['batch7', 'batch8', 'batch9', 'batch10']
        self.CELL_LINES = ['WT', 'TDP43', 'OPTN', 'TBK1', 'FUSHomozygous', 'SNCA']
        
class newNeuronsD8FigureConfig_UMAP2_WO_FUSRev_WO_FUSHet_with_SNCA_B7(newNeuronsD8FigureConfig_UMAP2_WO_FUSRev_WO_FUSHet_WithSNCA):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch7']

class newNeuronsD8FigureConfig_UMAP2_WO_FUSRev_WO_FUSHet_with_SNCA_B8(newNeuronsD8FigureConfig_UMAP2_WO_FUSRev_WO_FUSHet_WithSNCA):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch8']

class newNeuronsD8FigureConfig_UMAP2_WO_FUSRev_WO_FUSHet_with_SNCA_B9(newNeuronsD8FigureConfig_UMAP2_WO_FUSRev_WO_FUSHet_WithSNCA):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ['batch9']

class newNeuronsD8FigureConfig_UMAP2_WO_FUSRev_WO_FUSHet_with_SNCA_B10(newNeuronsD8FigureConfig_UMAP2_WO_FUSRev_WO_FUSHet_WithSNCA):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ['batch10']

# Only FUS lines
class newNeuronsD8FigureConfig_UMAP2_Only_FUSLines(FigureConfig):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3', 'batch7', 'batch8', 'batch9']

        self.EXPERIMENT_TYPE = 'neuronsDay8_new'
        self.CONDITIONS = ['Untreated']
        
        self.CELL_LINES = ['WT', 'FUSHomozygous', 'FUSHeterozygous', 'FUSRevertant']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False#True
        self.ADD_REP_TO_LABEL=False
        self.ADD_BATCH_TO_LABEL = False

class newNeuronsD8FigureConfig_UMAP2_Only_FUSLines_B1(newNeuronsD8FigureConfig_UMAP2_Only_FUSLines):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch1']

class newNeuronsD8FigureConfig_UMAP2_Only_FUSLines_B2(newNeuronsD8FigureConfig_UMAP2_Only_FUSLines):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch2']


class newNeuronsD8FigureConfig_UMAP2_Only_FUSLines_B3(newNeuronsD8FigureConfig_UMAP2_Only_FUSLines):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch3']

class newNeuronsD8FigureConfig_UMAP2_Only_FUSLines_B7(newNeuronsD8FigureConfig_UMAP2_Only_FUSLines):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch7']

class newNeuronsD8FigureConfig_UMAP2_Only_FUSLines_B8(newNeuronsD8FigureConfig_UMAP2_Only_FUSLines):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch8']

class newNeuronsD8FigureConfig_UMAP2_Only_FUSLines_B9(newNeuronsD8FigureConfig_UMAP2_Only_FUSLines):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ['batch9']

# Hits
class newNeuronsD8FigureConfig_UMAP2_Hits(FigureConfig):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3', 'batch7', 'batch8', 'batch9'] # without batch10

        self.EXPERIMENT_TYPE = 'neuronsDay8_new'
        self.CONDITIONS = ['Untreated']
        
        self.CELL_LINES = ['WT', 'TDP43', 'OPTN', 'TBK1', 'FUSRevertant', 'FUSHeterozygous', 'FUSHomozygous']
        
        self.MARKERS = ['Calreticulin', 'DCP1A', 'FUS', 'LAMP1', 'LSM14A', 'NCL', 'NEMO', 'NONO', 'PEX14', 'PML', 'PURA', 'SNCA', 'TIA1', 'TOMM20', 'Tubulin']
        # OLD self.MARKERS = ['HNRNPA1', 'NEMO', 'GM130', 'DCP1A', 'PURA', 'Calreticulin', 'Tubulin', 'PSD95', 'LAMP1', 'LSM14A', 'SNCA', 'PML', 'ANXA11', 'TIA1', 'NONO', 'CLTC', 'TOMM20', 'NCL', 'mitotracker', 'PEX14', 'FUS']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False#True
        self.ADD_REP_TO_LABEL=False
        self.ADD_BATCH_TO_LABEL = False

class newNeuronsD8FigureConfig_UMAP2_Hits_B1(newNeuronsD8FigureConfig_UMAP2_Hits):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch1']

class newNeuronsD8FigureConfig_UMAP2_Hits_B2(newNeuronsD8FigureConfig_UMAP2_Hits):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch2']

class newNeuronsD8FigureConfig_UMAP2_Hits_B3(newNeuronsD8FigureConfig_UMAP2_Hits):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch3']

class newNeuronsD8FigureConfig_UMAP2_Hits_B7(newNeuronsD8FigureConfig_UMAP2_Hits):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch7']

class newNeuronsD8FigureConfig_UMAP2_Hits_B8(newNeuronsD8FigureConfig_UMAP2_Hits):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch8']

class newNeuronsD8FigureConfig_UMAP2_Hits_B9(newNeuronsD8FigureConfig_UMAP2_Hits):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch9']

class newNeuronsD8FigureConfig_UMAP2_Hits_B10(newNeuronsD8FigureConfig_UMAP2_Hits):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch10']


### UMAP2 without Hits
class newNeuronsD8FigureConfig_UMAP2_wo_Hits(FigureConfig):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3', 'batch7', 'batch8', 'batch9'] # without batch10

        self.EXPERIMENT_TYPE = 'neuronsDay8_new'
        self.CONDITIONS = ['Untreated']
        
        self.CELL_LINES = ['WT', 'TDP43', 'OPTN', 'TBK1', 'FUSRevertant', 'FUSHeterozygous', 'FUSHomozygous']
        self.MARKERS_TO_EXCLUDE = self.MARKERS_TO_EXCLUDE + ['Calreticulin', 'DCP1A', 'FUS', 'LAMP1', 'LSM14A', 'NCL', 'NEMO', 'NONO', 'PEX14', 'PML', 'PURA', 'SNCA', 'TIA1', 'TOMM20', 'Tubulin']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False#True
        self.ADD_REP_TO_LABEL=False
        self.ADD_BATCH_TO_LABEL = False

class newNeuronsD8FigureConfig_UMAP2_wo_Hits_B1(newNeuronsD8FigureConfig_UMAP2_wo_Hits):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch1']

class newNeuronsD8FigureConfig_UMAP2_wo_Hits_B2(newNeuronsD8FigureConfig_UMAP2_wo_Hits):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch2']

class newNeuronsD8FigureConfig_UMAP2_wo_Hits_B3(newNeuronsD8FigureConfig_UMAP2_wo_Hits):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch3']

class newNeuronsD8FigureConfig_UMAP2_wo_Hits_B7(newNeuronsD8FigureConfig_UMAP2_wo_Hits):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch7']

class newNeuronsD8FigureConfig_UMAP2_wo_Hits_B8(newNeuronsD8FigureConfig_UMAP2_wo_Hits):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch8']

class newNeuronsD8FigureConfig_UMAP2_wo_Hits_B9(newNeuronsD8FigureConfig_UMAP2_wo_Hits):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch9']

class newNeuronsD8FigureConfig_UMAP2_wo_Hits_B10(newNeuronsD8FigureConfig_UMAP2_wo_Hits):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch10']


### FUS LINES
class newNeuronsD8FigureConfig_UMAP2_FUSLines(FigureConfig):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3', 'batch7', 'batch8', 'batch9', 'batch10']

        self.EXPERIMENT_TYPE = 'neuronsDay8_new'
        self.CONDITIONS = ['Untreated']
        
        self.CELL_LINES = ['WT', 'FUSRevertant', 'FUSHeterozygous', 'FUSHomozygous']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False#True
        self.ADD_REP_TO_LABEL=False
        self.ADD_BATCH_TO_LABEL = False

class newNeuronsD8FigureConfig_UMAP2_FUSLines_B1_wo_FUSMarker(newNeuronsD8FigureConfig_UMAP2_FUSLines):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch1']
        
        self.MARKERS_TO_EXCLUDE = self.MARKERS_TO_EXCLUDE + ['FUS']

class newNeuronsD8FigureConfig_UMAP2_FUSLines_B2_wo_FUSMarker(newNeuronsD8FigureConfig_UMAP2_FUSLines):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch2']
        
        self.MARKERS_TO_EXCLUDE = self.MARKERS_TO_EXCLUDE + ['FUS']

class newNeuronsD8FigureConfig_UMAP2_FUSLines_B3_wo_FUSMarker(newNeuronsD8FigureConfig_UMAP2_FUSLines):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch3']
        
        self.MARKERS_TO_EXCLUDE = self.MARKERS_TO_EXCLUDE + ['FUS']

class newNeuronsD8FigureConfig_UMAP2_FUSLines_B7_wo_FUSMarker(newNeuronsD8FigureConfig_UMAP2_FUSLines):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch7']
        
        self.MARKERS_TO_EXCLUDE = self.MARKERS_TO_EXCLUDE + ['FUS']

class newNeuronsD8FigureConfig_UMAP2_FUSLines_B8_wo_FUSMarker(newNeuronsD8FigureConfig_UMAP2_FUSLines):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch8']
        
        self.MARKERS_TO_EXCLUDE = self.MARKERS_TO_EXCLUDE + ['FUS']

class newNeuronsD8FigureConfig_UMAP2_FUSLines_B9_wo_FUSMarker(newNeuronsD8FigureConfig_UMAP2_FUSLines):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch9']
        
        self.MARKERS_TO_EXCLUDE = self.MARKERS_TO_EXCLUDE + ['FUS']

class newNeuronsD8FigureConfig_UMAP2_FUSLines_B10_wo_FUSMarker(newNeuronsD8FigureConfig_UMAP2_FUSLines):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch10']
        
        self.MARKERS_TO_EXCLUDE = self.MARKERS_TO_EXCLUDE + ['FUS']


class newNeuronsD8FigureConfig_UMAP2_CellPainting(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3', 'batch7', 'batch8', 'batch9', 'batch10']

        self.EXPERIMENT_TYPE = 'neuronsDay8_new'
        self.CONDITIONS = ['Untreated']
        self.MARKERS = ['DAPI', 'Calreticulin', 'NCL', 'mitotracker', 'Phalloidin', 'GM130']
        
        self.CELL_LINES = ['WT', 'TDP43', 'OPTN', 'TBK1', 'FUSRevertant', 'FUSHeterozygous', 'FUSHomozygous']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False#True
        self.ADD_REP_TO_LABEL=False
        self.ADD_BATCH_TO_LABEL = False

class newNeuronsD8FigureConfig_UMAP2_CellPainting_B1(newNeuronsD8FigureConfig_UMAP2_CellPainting):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ['batch1']

class newNeuronsD8FigureConfig_UMAP2_CellPainting_B2(newNeuronsD8FigureConfig_UMAP2_CellPainting):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ['batch2']

class newNeuronsD8FigureConfig_UMAP2_CellPainting_B3(newNeuronsD8FigureConfig_UMAP2_CellPainting):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ['batch3']

class newNeuronsD8FigureConfig_UMAP2_CellPainting_B7(newNeuronsD8FigureConfig_UMAP2_CellPainting):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ['batch7']

class newNeuronsD8FigureConfig_UMAP2_CellPainting_B8(newNeuronsD8FigureConfig_UMAP2_CellPainting):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ['batch8']

class newNeuronsD8FigureConfig_UMAP2_CellPainting_B9(newNeuronsD8FigureConfig_UMAP2_CellPainting):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ['batch9']

class newNeuronsD8FigureConfig_UMAP2_CellPainting_B10(newNeuronsD8FigureConfig_UMAP2_CellPainting):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ['batch10']

## Effect size

class NeuronsEffectsFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [f"batch{i}" for i in [1,2,3,7,8,9]]
        
        self.EXPERIMENT_TYPE = 'neuronsDay8_new'
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.BASELINE_PERTURB = {'WT_Untreated': 
                                 [f'{cell_line}_Untreated' for cell_line in 
                                  ['FUSHeterozygous','FUSHomozygous','FUSRevertant','OPTN','TDP43','TBK1']],
                                 'FUSRevertant_Untreated':
                                 [f'{cell_line}_Untreated' for cell_line in 
                                  ['FUSHeterozygous','FUSHomozygous','WT']]}

class NeuronsEffectsFigureConfig_SNCA(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [f"batch{i}" for i in [8,9]]
        
        self.EXPERIMENT_TYPE = 'neuronsDay8_new'
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.BASELINE_PERTURB = {'WT_Untreated': ['SNCA_Untreated'],
                                 'FUSRevertant_Untreated': ['SNCA_Untreated']}
        
class NeuronsMultiplexedEffectsFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [f"batch{i}" for i in [1,2,3,7,8,9]]
        
        self.EXPERIMENT_TYPE = 'neuronsDay8_new'
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = False
        self.BASELINE_PERTURB = {'WT_Untreated': 
                                 [f'{cell_line}_Untreated' for cell_line in 
                                  ['FUSHeterozygous','FUSHomozygous','FUSRevertant','OPTN','TDP43','TBK1']]}

class NeuronsMultiplexedEffectsFigureConfig_withSNCA(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [f"batch{i}" for i in [8,9]]
        
        self.EXPERIMENT_TYPE = 'neuronsDay8_new'
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = False
        self.BASELINE_PERTURB = {'WT_Untreated': 
                                 [f'{cell_line}_Untreated' for cell_line in 
                                  ['FUSHeterozygous','FUSHomozygous','FUSRevertant','OPTN','TDP43','TBK1', 'SNCA']]}

##########
# New dNLS
###########

class newDNLSUntreatedUMAP1DatasetConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3', 'batch4', 'batch5', 'batch6']
        
        self.MARKERS_TO_EXCLUDE = []
        self.EXPERIMENT_TYPE = 'dNLS'
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']

        self.SHOW_ARI = None#True
        self.ADD_REP_TO_LABEL=False
        self.ADD_BATCH_TO_LABEL = False
        self.ARI_LABELS_FUNC = MapLabelsFunction.MARKERS.name

class newDNLSUntreatedUMAP1DatasetConfig_B1(newDNLSUntreatedUMAP1DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch1"]

class newDNLSUntreatedUMAP1DatasetConfig_B2(newDNLSUntreatedUMAP1DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch2"]


class newDNLSUntreatedUMAP1DatasetConfig_B4(newDNLSUntreatedUMAP1DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch4"]

class newDNLSUntreatedUMAP1DatasetConfig_B5(newDNLSUntreatedUMAP1DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch5"]

class newDNLSUntreatedUMAP1DatasetConfig_B6(newDNLSUntreatedUMAP1DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch6"]

## UMAP 0

class newDNLSUMAP0DatasetConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = None
        
        self.MARKERS_TO_EXCLUDE = []
        self.EXPERIMENT_TYPE = 'dNLS'
        self.CELL_LINES = ['dNLS']
        self.SHOW_ARI = False #True
        self.ADD_REP_TO_LABEL=False #True
        self.ADD_BATCH_TO_LABEL = False
        self.ARI_LABELS_FUNC = MapLabelsFunction.CELL_LINES_CONDITIONS.name
        
class newDNLSUMAP0DatasetConfig_Hits(newDNLSUMAP0DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch1", "batch2", "batch3", "batch4", "batch5", "batch6"]
        self.MARKERS = ['TDP43', 'DCP1A', 'LSM14A']

class newDNLSUMAP0DatasetConfig_TDP43_woB3(newDNLSUMAP0DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch1", "batch2", "batch4", "batch5", "batch6"]
        self.MARKERS = ['TDP43']

class newDNLSUMAP0DatasetConfig_Hits_With_WT(newDNLSUMAP0DatasetConfig_Hits):
    def __init__(self):
        super().__init__()
    
        self.CELL_LINES = None

class newDNLSUMAP0B1DatasetConfig(newDNLSUMAP0DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch1"]

class newDNLSUMAP0B1DatasetConfig_TDP43(newDNLSUMAP0B1DatasetConfig):
    def __init__(self):
        super().__init__()

        self.MARKERS = ['TDP43']

class newDNLSUMAP0B2DatasetConfig(newDNLSUMAP0DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch2"]

class newDNLSUMAP0B2DatasetConfig_TDP43(newDNLSUMAP0B2DatasetConfig):
    def __init__(self):
        super().__init__()

        self.MARKERS = ['TDP43']

class newDNLSUMAP0B4DatasetConfig(newDNLSUMAP0DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch4"]

class newDNLSUMAP0B4DatasetConfig_TDP43(newDNLSUMAP0B4DatasetConfig):
    def __init__(self):
        super().__init__()

        self.MARKERS = ['TDP43']

class newDNLSUMAP0B5DatasetConfig(newDNLSUMAP0DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch5"]


class newDNLSUMAP0B5DatasetConfig_TDP43(newDNLSUMAP0B5DatasetConfig):
    def __init__(self):
        super().__init__()

        self.MARKERS = ['TDP43']

class newDNLSUMAP0B6DatasetConfig(newDNLSUMAP0DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch6"]

class newDNLSUMAP0B6DatasetConfig_TDP43(newDNLSUMAP0B6DatasetConfig):
    def __init__(self):
        super().__init__()

        self.MARKERS = ['TDP43']

class newDNLSUMAP0DatasetConfig_WithWT(newDNLSUMAP0DatasetConfig):
    def __init__(self):
        super().__init__()

        self.CELL_LINES = []

        
class newDNLSUMAP0B1DatasetConfig_WithWT(newDNLSUMAP0DatasetConfig_WithWT):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch1"]

class newDNLSUMAP0B2DatasetConfig_WithWT(newDNLSUMAP0DatasetConfig_WithWT):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch2"]

class newDNLSUMAP0B3DatasetConfig_WithWT(newDNLSUMAP0DatasetConfig_WithWT):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch3"]

class newDNLSUMAP0B4DatasetConfig_WithWT(newDNLSUMAP0DatasetConfig_WithWT):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch4"]

class newDNLSUMAP0B5DatasetConfig_WithWT(newDNLSUMAP0DatasetConfig_WithWT):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch5"]

class newDNLSUMAP0B6DatasetConfig_WithWT(newDNLSUMAP0DatasetConfig_WithWT):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ["batch6"]

# UMAP2

class newDNLSFigureConfig_UMAP2(FigureConfig):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch4', 'batch5', 'batch6']
        
        self.EXPERIMENT_TYPE = 'dNLS' #'dNLS'  
        self.CELL_LINES = ['dNLS']
        
        # Decide if to show ARI metric on the UMAP
        self.SHOW_ARI = False#True
        self.ADD_REP_TO_LABEL=False
        self.ADD_BATCH_TO_LABEL = False

class newDNLSFigureConfig_UMAP2_B1(newDNLSFigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch1']

class newDNLSFigureConfig_UMAP2_B2(newDNLSFigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch2']

class newDNLSFigureConfig_UMAP2_B4(newDNLSFigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch4']

class newDNLSFigureConfig_UMAP2_B5(newDNLSFigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch5']

class newDNLSFigureConfig_UMAP2_B6(newDNLSFigureConfig_UMAP2):
    def __init__(self):
        super().__init__()

         
        self.INPUT_FOLDERS = ['batch6']


## Effect size

class dNLSEffectsFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [f"batch{i}" for i in [1,2,4,5,6]]

        self.EXPERIMENT_TYPE = 'dNLS'
        self.MARKERS = list(PlotConfig().COLOR_MAPPINGS_MARKERS.keys())
        self.BASELINE_PERTURB = {'dNLS_Untreated':['dNLS_DOX']}

        self.CELL_LINES = ['dNLS']

class dNLSEffectsFigureConfig_Multiplexed(dNLSEffectsFigureConfig):
    def __init__(self):
        super().__init__()

        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = False

############
# NIH
############

# UMAP1
class NIH_UMAP1_DatasetConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3']
        
        self.EXPERIMENT_TYPE = 'NIH'
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['CD41']

        self.SHOW_ARI = False
        self.ADD_REP_TO_LABEL=False
        self.ADD_BATCH_TO_LABEL = False
        self.ARI_LABELS_FUNC = MapLabelsFunction.MARKERS.name

class NIH_UMAP1_DatasetConfig_B1(NIH_UMAP1_DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch1"]

class NIH_UMAP1_DatasetConfig_B2(NIH_UMAP1_DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch2"]

class NIH_UMAP1_DatasetConfig_B3(NIH_UMAP1_DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch3"]

# UMAP0 Stress
class NIH_UMAP0_Stress_DatasetConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3']
        
        self.EXPERIMENT_TYPE = 'NIH'
        self.CELL_LINES = ['WT']
        self.CONDITIONS = None

        self.SHOW_ARI = False
        self.ADD_REP_TO_LABEL=False
        self.ADD_BATCH_TO_LABEL = False
        self.ARI_LABELS_FUNC = MapLabelsFunction.MARKERS.name


class NIH_UMAP0_Stress_DatasetConfig_Subset(NIH_UMAP0_Stress_DatasetConfig):
    def __init__(self):
        super().__init__()

        self.MARKERS = ['G3BP1', 'TIA1', 'FMRP', 'PURA']

class NIH_UMAP0_Stress_DatasetConfig_B1(NIH_UMAP0_Stress_DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch1"]

class NIH_UMAP0_Stress_DatasetConfig_B2(NIH_UMAP0_Stress_DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch2"]

class NIH_UMAP0_Stress_DatasetConfig_B3(NIH_UMAP0_Stress_DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch3"]

## Only positive controls
class NIH_UMAP0_Stress_DatasetConfig_PositiveControls(NIH_UMAP0_Stress_DatasetConfig):
    def __init__(self):
        super().__init__()

        self.MARKERS = ['G3BP1', 'FMRP', 'PURA']

class NIH_UMAP0_Stress_DatasetConfig_PositiveControls_B1(NIH_UMAP0_Stress_DatasetConfig_PositiveControls):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch1"]

class NIH_UMAP0_Stress_DatasetConfig_PositiveControls_B2(NIH_UMAP0_Stress_DatasetConfig_PositiveControls):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch2"]

class NIH_UMAP0_Stress_DatasetConfig_PositiveControls_B3(NIH_UMAP0_Stress_DatasetConfig_PositiveControls):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch3"]

# UMAP0 FUS
class NIH_UMAP0_FUS_DatasetConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3']
        
        self.EXPERIMENT_TYPE = 'NIH'
        self.CELL_LINES = None
        self.CONDITIONS = ['Untreated']

        self.SHOW_ARI = False
        self.ADD_REP_TO_LABEL=False
        self.ADD_BATCH_TO_LABEL = False
        self.ARI_LABELS_FUNC = MapLabelsFunction.MARKERS.name

class NIH_UMAP0_FUS_DatasetConfig_B1(NIH_UMAP0_FUS_DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch1"]

class NIH_UMAP0_FUS_DatasetConfig_B2(NIH_UMAP0_FUS_DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch2"]

class NIH_UMAP0_FUS_DatasetConfig_B3(NIH_UMAP0_FUS_DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch3"]

# UMAP2 FUS
class NIH_UMAP2_FUS_DatasetConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3']
        
        self.EXPERIMENT_TYPE = 'NIH'
        self.CELL_LINES = None
        self.CONDITIONS = ['Untreated']
        self.MARKERS_TO_EXCLUDE = ['CD41']

        self.SHOW_ARI = False
        self.ADD_REP_TO_LABEL=False
        self.ADD_BATCH_TO_LABEL = False
        self.ARI_LABELS_FUNC = MapLabelsFunction.MARKERS.name

class NIH_UMAP2_FUS_DatasetConfig_B1(NIH_UMAP2_FUS_DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch1"]

class NIH_UMAP2_FUS_DatasetConfig_B2(NIH_UMAP2_FUS_DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch2"]

class NIH_UMAP2_FUS_DatasetConfig_B3(NIH_UMAP2_FUS_DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch3"]

class NIH_UMAP2_FUS_DatasetConfig_wo_FUS_Marker(NIH_UMAP2_FUS_DatasetConfig):
    def __init__(self):
        super().__init__()

        self.MARKERS_TO_EXCLUDE = self.MARKERS_TO_EXCLUDE + ['FUS']

class NIH_UMAP2_FUS_DatasetConfig_wo_FUS_Marker_B1(NIH_UMAP2_FUS_DatasetConfig_wo_FUS_Marker):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch1"]

class NIH_UMAP2_FUS_DatasetConfig_wo_FUS_Marker_B2(NIH_UMAP2_FUS_DatasetConfig_wo_FUS_Marker):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch2"]

class NIH_UMAP2_FUS_DatasetConfig_wo_FUS_Marker_B3(NIH_UMAP2_FUS_DatasetConfig_wo_FUS_Marker):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch3"]

# UMAP2 Stress
class NIH_UMAP2_Stress_DatasetConfig(FigureConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = ['batch1', 'batch2', 'batch3']
        
        self.EXPERIMENT_TYPE = 'NIH'
        self.CELL_LINES = ['WT']
        self.CONDITIONS = None
        self.MARKERS_TO_EXCLUDE = ['CD41']

        self.SHOW_ARI = False
        self.ADD_REP_TO_LABEL=False
        self.ADD_BATCH_TO_LABEL = False
        self.ARI_LABELS_FUNC = MapLabelsFunction.MARKERS.name

class NIH_UMAP2_Stress_DatasetConfig_B1(NIH_UMAP2_Stress_DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch1"]

class NIH_UMAP2_Stress_DatasetConfig_B2(NIH_UMAP2_Stress_DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch2"]

class NIH_UMAP2_Stress_DatasetConfig_B3(NIH_UMAP2_Stress_DatasetConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = ["batch3"]

# Effect size
class NIHEffectsFigureConfig(FigureConfig):
    def __init__(self):
        super().__init__()
        self.INPUT_FOLDERS = [f"batch{i}" for i in [1,2,3]]
        
        self.EXPERIMENT_TYPE = 'NIH'
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = True
        self.BASELINE_PERTURB = {'WT_Untreated': ['WT_stress']}


class NIHEffectsFigureConfig_Multiplexed(NIHEffectsFigureConfig):
    def __init__(self):
        super().__init__()
        self.ADD_BATCH_TO_LABEL = True
        self.ADD_REP_TO_LABEL = False