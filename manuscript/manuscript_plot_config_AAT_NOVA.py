import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from manuscript.plot_config import PlotConfig
from manuscript.manuscript_plot_config import *
from src.datasets.label_utils import MapLabelsFunction

class AAT_NOVA_BaseFigureConfig(PlotConfig):
    def __init__(self):
        super().__init__()
        # Markers
        self.COLOR_MAPPINGS_AAT_NOVA = {
            'DAPI': {self.MAPPINGS_ALIAS_KEY: 'Nucleus', self.MAPPINGS_COLOR_KEY: '#AFBDFF'},
            'Cas3': {self.MAPPINGS_ALIAS_KEY: 'Cas3', self.MAPPINGS_COLOR_KEY: '#4343FE'},
            'FK-2': {self.MAPPINGS_ALIAS_KEY: 'FK-2', self.MAPPINGS_COLOR_KEY: '#921010'},
            'SMI32': {self.MAPPINGS_ALIAS_KEY: 'SMI32', self.MAPPINGS_COLOR_KEY: '#12F986'},
            'pDRP1': {self.MAPPINGS_ALIAS_KEY: 'pDRP1', self.MAPPINGS_COLOR_KEY: '#66CDAA'},
            'TOMM20': {self.MAPPINGS_ALIAS_KEY: 'TOMM20', self.MAPPINGS_COLOR_KEY: '#18E4CF'},
            'pCaMKIIa': {self.MAPPINGS_ALIAS_KEY: 'pCaMKIIa', self.MAPPINGS_COLOR_KEY: '#168FB2'},
            'pTDP-43': {self.MAPPINGS_ALIAS_KEY: 'pTDP-43', self.MAPPINGS_COLOR_KEY: '#A80358'},
            'TDP-43': {self.MAPPINGS_ALIAS_KEY: 'TDP-43', self.MAPPINGS_COLOR_KEY: '#9968CB'},
            'ATF6': {self.MAPPINGS_ALIAS_KEY: 'ATF6', self.MAPPINGS_COLOR_KEY: '#D257EA'},
            'pAMPK': {self.MAPPINGS_ALIAS_KEY: 'pAMPK', self.MAPPINGS_COLOR_KEY: '#E6A9EA'},
            'HDGFL2': {self.MAPPINGS_ALIAS_KEY: 'HDGFL2', self.MAPPINGS_COLOR_KEY: '#F04521'},
            'pS6': {self.MAPPINGS_ALIAS_KEY: 'pS6', self.MAPPINGS_COLOR_KEY: '#F08F21'},
            'PAR': {self.MAPPINGS_ALIAS_KEY: 'PAR', self.MAPPINGS_COLOR_KEY: '#F1CBDD'},
            'UNC13A': {self.MAPPINGS_ALIAS_KEY: 'UNC13A', self.MAPPINGS_COLOR_KEY: '#37378D'},
            'Calreticulin': {self.MAPPINGS_ALIAS_KEY: 'Calreticulin', self.MAPPINGS_COLOR_KEY: 'gray'},
            'LC3-II': {self.MAPPINGS_ALIAS_KEY: 'LC3-II', self.MAPPINGS_COLOR_KEY: '#DEDB23'},
            'p62': {self.MAPPINGS_ALIAS_KEY: 'p62', self.MAPPINGS_COLOR_KEY: '#AF8215'},
            'CathepsinD': {self.MAPPINGS_ALIAS_KEY: 'CathepsinD', self.MAPPINGS_COLOR_KEY: '#32AC0E'},
        }

        # conditions
        self.COLOR_MAPPINGS_AAT_NOVA_CONDITIONS = {
            'PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'PPP2R1A', self.MAPPINGS_COLOR_KEY: '#AFBDFF'},
            'HMGCS1': {self.MAPPINGS_ALIAS_KEY: 'HMGCS1', self.MAPPINGS_COLOR_KEY: '#4343FE'},
            'PIK3C3': {self.MAPPINGS_ALIAS_KEY: 'PIK3C3', self.MAPPINGS_COLOR_KEY: '#921010'},
            'NDUFAB1': {self.MAPPINGS_ALIAS_KEY: 'NDUFAB1', self.MAPPINGS_COLOR_KEY: '#12F986'},
            'MAPKAP1': {self.MAPPINGS_ALIAS_KEY: 'MAPKAP1', self.MAPPINGS_COLOR_KEY: '#66CDAA'},
            'NDUFS2': {self.MAPPINGS_ALIAS_KEY: 'NDUFS2', self.MAPPINGS_COLOR_KEY: '#18E4CF'},
            'RALA': {self.MAPPINGS_ALIAS_KEY: 'RALA', self.MAPPINGS_COLOR_KEY: '#168FB2'},
            'TLK1': {self.MAPPINGS_ALIAS_KEY: 'TLK1', self.MAPPINGS_COLOR_KEY: '#A80358'},
            'NRIP1': {self.MAPPINGS_ALIAS_KEY: 'NRIP1', self.MAPPINGS_COLOR_KEY: '#9968CB'},
            'TARDBP': {self.MAPPINGS_ALIAS_KEY: 'TARDBP', self.MAPPINGS_COLOR_KEY: '#D257EA'},
            'RANBP17': {self.MAPPINGS_ALIAS_KEY: 'RANBP17', self.MAPPINGS_COLOR_KEY: '#E6A9EA'},
            'CYLD': {self.MAPPINGS_ALIAS_KEY: 'CYLD', self.MAPPINGS_COLOR_KEY: '#F04521'},
            'NT-1873': {self.MAPPINGS_ALIAS_KEY: 'NT-1873', self.MAPPINGS_COLOR_KEY: '#F08F21'},
            'NT-6301-3085': {self.MAPPINGS_ALIAS_KEY: 'NT-6301-3085', self.MAPPINGS_COLOR_KEY: '#F1CBDD'},
            'Intergenic': {self.MAPPINGS_ALIAS_KEY: 'Intergenic', self.MAPPINGS_COLOR_KEY: '#37378D'},
            'Untreated': {self.MAPPINGS_ALIAS_KEY: 'Untreated', self.MAPPINGS_COLOR_KEY: 'gray'},
        }

        # Conditions per cell line
        self.COLOR_MAPPINGS_AAT_NOVA_CONDITIONS_PER_CELL_LINE = {
            'CTL_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'CTL_PPP2R1A', self.MAPPINGS_COLOR_KEY: '#C9D5FF'}, # brighter
            'C9_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'C9_PPP2R1A', self.MAPPINGS_COLOR_KEY: '#8A9BDB'}, # darker

            'CTL_HMGCS1': {self.MAPPINGS_ALIAS_KEY: 'CTL_HMGCS1', self.MAPPINGS_COLOR_KEY: '#6A6AFF'},
            'C9_HMGCS1': {self.MAPPINGS_ALIAS_KEY: 'C9_HMGCS1', self.MAPPINGS_COLOR_KEY: '#2E2EB2'},

            'CTL_PIK3C3': {self.MAPPINGS_ALIAS_KEY: 'CTL_PIK3C3', self.MAPPINGS_COLOR_KEY: '#B83232'},
            'C9_PIK3C3': {self.MAPPINGS_ALIAS_KEY: 'C9_PIK3C3', self.MAPPINGS_COLOR_KEY: '#6E0C0C'},

            'CTL_NDUFAB1': {self.MAPPINGS_ALIAS_KEY: 'CTL_NDUFAB1', self.MAPPINGS_COLOR_KEY: '#4AFFA5'},
            'C9_NDUFAB1': {self.MAPPINGS_ALIAS_KEY: 'C9_NDUFAB1', self.MAPPINGS_COLOR_KEY: '#0E9E5F'},

            'CTL_MAPKAP1': {self.MAPPINGS_ALIAS_KEY: 'CTL_MAPKAP1', self.MAPPINGS_COLOR_KEY: '#88E0C0'},
            'C9_MAPKAP1': {self.MAPPINGS_ALIAS_KEY: 'C9_MAPKAP1', self.MAPPINGS_COLOR_KEY: '#3E8E76'},

            'CTL_NDUFS2': {self.MAPPINGS_ALIAS_KEY: 'CTL_NDUFS2', self.MAPPINGS_COLOR_KEY: '#4AF8E2'},
            'C9_NDUFS2': {self.MAPPINGS_ALIAS_KEY: 'C9_NDUFS2', self.MAPPINGS_COLOR_KEY: '#0E9989'},

            'CTL_RALA': {self.MAPPINGS_ALIAS_KEY: 'CTL_RALA', self.MAPPINGS_COLOR_KEY: '#3AB6D6'},
            'C9_RALA': {self.MAPPINGS_ALIAS_KEY: 'C9_RALA', self.MAPPINGS_COLOR_KEY: '#0C6479'},

            'CTL_TLK1': {self.MAPPINGS_ALIAS_KEY: 'CTL_TLK1', self.MAPPINGS_COLOR_KEY: '#D92678'},
            'C9_TLK1': {self.MAPPINGS_ALIAS_KEY: 'C9_TLK1', self.MAPPINGS_COLOR_KEY: '#6A0231'},

            'CTL_NRIP1': {self.MAPPINGS_ALIAS_KEY: 'CTL_NRIP1', self.MAPPINGS_COLOR_KEY: '#B488E0'},
            'C9_NRIP1': {self.MAPPINGS_ALIAS_KEY: 'C9_NRIP1', self.MAPPINGS_COLOR_KEY: '#62307E'},

            'CTL_TARDBP': {self.MAPPINGS_ALIAS_KEY: 'CTL_TARDBP', self.MAPPINGS_COLOR_KEY: '#E680F5'},
            'C9_TARDBP': {self.MAPPINGS_ALIAS_KEY: 'C9_TARDBP', self.MAPPINGS_COLOR_KEY: '#8615A3'},

            'CTL_RANBP17': {self.MAPPINGS_ALIAS_KEY: 'CTL_RANBP17', self.MAPPINGS_COLOR_KEY: '#F4C5F4'},
            'C9_RANBP17': {self.MAPPINGS_ALIAS_KEY: 'C9_RANBP17', self.MAPPINGS_COLOR_KEY: '#914691'},

            'CTL_CYLD': {self.MAPPINGS_ALIAS_KEY: 'CTL_CYLD', self.MAPPINGS_COLOR_KEY: '#FF6F4A'},
            'C9_CYLD': {self.MAPPINGS_ALIAS_KEY: 'C9_CYLD', self.MAPPINGS_COLOR_KEY: '#91260E'},

            'CTL_NT-1873': {self.MAPPINGS_ALIAS_KEY: 'CTL_NT-1873', self.MAPPINGS_COLOR_KEY: '#FFB366'},
            'C9_NT-1873': {self.MAPPINGS_ALIAS_KEY: 'C9_NT-1873', self.MAPPINGS_COLOR_KEY: '#9A4E08'},

            'CTL_NT-6301-3085': {self.MAPPINGS_ALIAS_KEY: 'CTL_NT-6301-3085', self.MAPPINGS_COLOR_KEY: '#F9E1EC'},
            'C9_NT-6301-3085': {self.MAPPINGS_ALIAS_KEY: 'C9_NT-6301-3085', self.MAPPINGS_COLOR_KEY: '#98506B'},

            'CTL_Intergenic': {self.MAPPINGS_ALIAS_KEY: 'CTL_Intergenic', self.MAPPINGS_COLOR_KEY: '#5C5CC0'},
            'C9_Intergenic': {self.MAPPINGS_ALIAS_KEY: 'C9_Intergenic', self.MAPPINGS_COLOR_KEY: '#1C1C5C'},

            'CTL_Untreated': {self.MAPPINGS_ALIAS_KEY: 'CTL_Untreated', self.MAPPINGS_COLOR_KEY: '#B0B0B0'},
            'C9_Untreated': {self.MAPPINGS_ALIAS_KEY: 'C9_Untreated', self.MAPPINGS_COLOR_KEY: '#505050'},
        }


        # marker per condition
        self.COLOR_MAPPINGS_AAT_NOVA_MARKER_PER_CONDITIONS = {
            'DAPI_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Nucleus_Untreated', self.MAPPINGS_COLOR_KEY: '#000000'},
            'Cas3_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Cas3_Untreated', self.MAPPINGS_COLOR_KEY: '#000000'},
            'FK-2': {self.MAPPINGS_ALIAS_KEY: 'FK-2_Untreated', self.MAPPINGS_COLOR_KEY: '#000000'},
            'SMI32_Untreated': {self.MAPPINGS_ALIAS_KEY: 'SMI32_Untreated', self.MAPPINGS_COLOR_KEY: '#000000'},
            'pDRP1_Untreated': {self.MAPPINGS_ALIAS_KEY: 'pDRP1_Untreated', self.MAPPINGS_COLOR_KEY: '#000000'},
            'TOMM20_Untreated': {self.MAPPINGS_ALIAS_KEY: 'TOMM20_Untreated', self.MAPPINGS_COLOR_KEY: '#000000'},
            'pCaMKIIa_Untreated': {self.MAPPINGS_ALIAS_KEY: 'pCaMKIIa_Untreated', self.MAPPINGS_COLOR_KEY: '#000000'},
            'pTDP-43_Untreated': {self.MAPPINGS_ALIAS_KEY: 'pTDP-43_Untreated', self.MAPPINGS_COLOR_KEY: '#000000'},
            'TDP-43_Untreated': {self.MAPPINGS_ALIAS_KEY: 'TDP-43_Untreated', self.MAPPINGS_COLOR_KEY: '#000000'},
            'ATF6_Untreated': {self.MAPPINGS_ALIAS_KEY: 'ATF6_Untreated', self.MAPPINGS_COLOR_KEY: '#000000'},
            'pAMPK_Untreated': {self.MAPPINGS_ALIAS_KEY: 'pAMPK_Untreated', self.MAPPINGS_COLOR_KEY: '#000000'},
            'HDGFL2_Untreated': {self.MAPPINGS_ALIAS_KEY: 'HDGFL2_Untreated', self.MAPPINGS_COLOR_KEY: '#000000'},
            'pS6_Untreated': {self.MAPPINGS_ALIAS_KEY: 'pS6_Untreated', self.MAPPINGS_COLOR_KEY: '#000000'},
            'PAR_Untreated': {self.MAPPINGS_ALIAS_KEY: 'PAR_Untreated', self.MAPPINGS_COLOR_KEY: '#000000'},
            'UNC13A_Untreated': {self.MAPPINGS_ALIAS_KEY: 'UNC13A_Untreated', self.MAPPINGS_COLOR_KEY: '#000000'},
            'Calreticulin_Untreated': {self.MAPPINGS_ALIAS_KEY: 'Calreticulin_Untreated', self.MAPPINGS_COLOR_KEY: '#000000'},
            'LC3-II_Untreated': {self.MAPPINGS_ALIAS_KEY: 'LC3-II_Untreated', self.MAPPINGS_COLOR_KEY: '#000000'},
            'p62_Untreated': {self.MAPPINGS_ALIAS_KEY: 'p62_Untreated', self.MAPPINGS_COLOR_KEY: '#000000'},
            'CathepsinD_Untreated': {self.MAPPINGS_ALIAS_KEY: 'CathepsinD_Untreated', self.MAPPINGS_COLOR_KEY: '#000000'},
            
            'DAPI_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'Nucleus_PPP2R1A', self.MAPPINGS_COLOR_KEY: '#AFBDFF'},
            'Cas3_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'Cas3_PPP2R1A', self.MAPPINGS_COLOR_KEY: '#4343FE'},
            'FK-2_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'FK-2_PPP2R1A', self.MAPPINGS_COLOR_KEY: '#921010'},
            'SMI32_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'SMI32_PPP2R1A', self.MAPPINGS_COLOR_KEY: '#12F986'},
            'pDRP1_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'pDRP1_PPP2R1A', self.MAPPINGS_COLOR_KEY: '#66CDAA'},
            'TOMM20_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'TOMM20_PPP2R1A', self.MAPPINGS_COLOR_KEY: '#18E4CF'},
            'pCaMKIIa_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'pCaMKIIa_PPP2R1A', self.MAPPINGS_COLOR_KEY: '#168FB2'},
            'pTDP-43_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'pTDP-43_PPP2R1A', self.MAPPINGS_COLOR_KEY: '#A80358'},
            'TDP-43_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'TDP-43_PPP2R1A', self.MAPPINGS_COLOR_KEY: '#9968CB'},
            'ATF6_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'ATF6_PPP2R1A', self.MAPPINGS_COLOR_KEY: '#D257EA'},
            'pAMPK_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'pAMPK_PPP2R1A', self.MAPPINGS_COLOR_KEY: '#E6A9EA'},
            'HDGFL2_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'HDGFL2_PPP2R1A', self.MAPPINGS_COLOR_KEY: '#F04521'},
            'pS6_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'pS6_PPP2R1A', self.MAPPINGS_COLOR_KEY: '#F08F21'},
            'PAR_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'PAR_PPP2R1A', self.MAPPINGS_COLOR_KEY: '#F1CBDD'},
            'UNC13A_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'UNC13A_PPP2R1A', self.MAPPINGS_COLOR_KEY: '#37378D'},
            'Calreticulin_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'Calreticulin_PPP2R1A', self.MAPPINGS_COLOR_KEY: 'gray'},
            'LC3-II_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'LC3-II_PPP2R1A', self.MAPPINGS_COLOR_KEY: '#DEDB23'},
            'p62_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'p62_PPP2R1A', self.MAPPINGS_COLOR_KEY: '#AF8215'},
            'CathepsinD_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'CathepsinD_PPP2R1A', self.MAPPINGS_COLOR_KEY: '#32AC0E'},
        }

        # Cell lines - 
        self.COLOR_MAPPINGS_AAT_NOVA_CELL_LINES: Dict[str, Dict[str, str]] = { 
            'C9': {self.MAPPINGS_ALIAS_KEY: 'C9', self.MAPPINGS_COLOR_KEY: '#1F77B4'}, 
            'CTL': {self.MAPPINGS_ALIAS_KEY: 'CTL', self.MAPPINGS_COLOR_KEY: '#2E8B57'}, 
            }

        self.COLOR_MAPPINGS_AAT_NOVA_CATEGORIES = {
            # Proteostasis (Blue)
            'Cas3': {self.MAPPINGS_ALIAS_KEY: 'Proteostasis', self.MAPPINGS_COLOR_KEY: '#1F77B4'},
            'FK-2': {self.MAPPINGS_ALIAS_KEY: 'Proteostasis', self.MAPPINGS_COLOR_KEY: '#1F77B4'},
            'pDRP1': {self.MAPPINGS_ALIAS_KEY: 'Proteostasis', self.MAPPINGS_COLOR_KEY: '#1F77B4'},
            'TOMM20': {self.MAPPINGS_ALIAS_KEY: 'Proteostasis', self.MAPPINGS_COLOR_KEY: '#1F77B4'},
            'pCaMKIIa': {self.MAPPINGS_ALIAS_KEY: 'Proteostasis', self.MAPPINGS_COLOR_KEY: '#1F77B4'},
            'pTDP-43': {self.MAPPINGS_ALIAS_KEY: 'Proteostasis', self.MAPPINGS_COLOR_KEY: '#1F77B4'},
            'TDP-43': {self.MAPPINGS_ALIAS_KEY: 'Proteostasis', self.MAPPINGS_COLOR_KEY: '#1F77B4'},
            'Protein-degradation': {self.MAPPINGS_ALIAS_KEY: 'Proteostasis', self.MAPPINGS_COLOR_KEY: '#1F77B4'},

            # Neuronal Cell Death / Senescence (Red)
            'ATF6': {self.MAPPINGS_ALIAS_KEY: 'Neuronal Cell Death/Senescence', self.MAPPINGS_COLOR_KEY: '#E41A1C'},
            'HDGFL2': {self.MAPPINGS_ALIAS_KEY: 'Neuronal Cell Death/Senescence', self.MAPPINGS_COLOR_KEY: '#E41A1C'},
            'Calreticulin': {self.MAPPINGS_ALIAS_KEY: 'Neuronal Cell Death/Senescence', self.MAPPINGS_COLOR_KEY: '#E41A1C'},
            'LC3-II': {self.MAPPINGS_ALIAS_KEY: 'Neuronal Cell Death/Senescence', self.MAPPINGS_COLOR_KEY: '#E41A1C'},
            'p62': {self.MAPPINGS_ALIAS_KEY: 'Neuronal Cell Death/Senescence', self.MAPPINGS_COLOR_KEY: '#E41A1C'},
            'CathepsinD': {self.MAPPINGS_ALIAS_KEY: 'Neuronal Cell Death/Senescence', self.MAPPINGS_COLOR_KEY: '#E41A1C'},

            # Synaptic and Neuronal Function (Green)
            'SMI32': {self.MAPPINGS_ALIAS_KEY: 'Synaptic and Neuronal Function', self.MAPPINGS_COLOR_KEY: '#238B45'},
            'Senescence-signaling': {self.MAPPINGS_ALIAS_KEY: 'Synaptic and Neuronal Function', self.MAPPINGS_COLOR_KEY: '#238B45'},

            # DNA and RNA Defects (Purple)
            'pAMPK': {self.MAPPINGS_ALIAS_KEY: 'DNA and RNA Defects', self.MAPPINGS_COLOR_KEY: '#6A3D9A'},
            'pS6': {self.MAPPINGS_ALIAS_KEY: 'DNA and RNA Defects', self.MAPPINGS_COLOR_KEY: '#6A3D9A'},
            'PAR': {self.MAPPINGS_ALIAS_KEY: 'DNA and RNA Defects', self.MAPPINGS_COLOR_KEY: '#6A3D9A'},
            'UNC13A': {self.MAPPINGS_ALIAS_KEY: 'DNA and RNA Defects', self.MAPPINGS_COLOR_KEY: '#6A3D9A'},

            # Pathological Protein Aggregation (Orange)
            'TDP-43': {self.MAPPINGS_ALIAS_KEY: 'Pathological Protein Aggregation', self.MAPPINGS_COLOR_KEY: '#FF7F0E'}
        }

        # cell line and conditions
        self.COLOR_MAPPINGS_ALS_CONDITIONS_AAT_NOVA: Dict[str, Dict[str, str]] = {
            # (muted blue)
            'C9': {self.MAPPINGS_ALIAS_KEY: 'C9 Untreated', self.MAPPINGS_COLOR_KEY: '#5271A5'},


            # (soft green)
            'CTL': {self.MAPPINGS_ALIAS_KEY: 'CTL Untreated', self.MAPPINGS_COLOR_KEY: '#6A9E6D'},

            # (warm orange)
            'C9_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'C9 PPP2R1A', self.MAPPINGS_COLOR_KEY: '#D08C60'},

            # (soft yellow)
            'CTL_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'CTL PPP2R1A', self.MAPPINGS_COLOR_KEY: '#E6D96A'},
       }

class UMAP1AAT_NOVAPlotConfig(AAT_NOVA_BaseFigureConfig):
    def __init__(self):
        super().__init__()
        
        self.ORDERED_MARKER_NAMES = ["DAPI", "Cas3", "FK-2", "SMI32", "pDRP1", "TOMM20", "pCaMKIIa", "pTDP-43", "TDP-43", "ATF6", "pAMPK", "HDGFL2", "pS6", "PAR", "UNC13A", "Calreticulin", "LC3-II", "p62", "CathepsinD"]

    
        # Set the size of the dots
        self.SIZE = 1
        self.ALPHA = 1
        # How labels are shown in legend
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.MARKERS.name
        # umap type
        self.UMAP_TYPE = 1
        # Colors 
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_AAT_NOVA
        self.COLOR_MAPPINGS_MARKERS = self.COLOR_MAPPINGS_AAT_NOVA

class UMAP1AAT_NOVAPlotConfigConditions(AAT_NOVA_BaseFigureConfig):
    def __init__(self):
        super().__init__()
        
        self.ORDERED_MARKER_NAMES = None
        # Set the size of the dots
        self.SIZE = 1
        self.ALPHA = 1
        # How labels are shown in legend
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.MARKERS_CONDITIONS.name
        # umap type
        self.UMAP_TYPE = 1
        # Colors 
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_AAT_NOVA_CONDITIONS
        self.COLOR_MAPPINGS_MARKERS = self.COLOR_MAPPINGS_AAT_NOVA_CONDITIONS

class UMAP1AAT_NOVAPPP2R1APlotConfig(AAT_NOVA_BaseFigureConfig):
    def __init__(self):
        super().__init__()    
        # Set the size of the dots
        self.SIZE = 1
        self.ALPHA = 1
        # How labels are shown in legend
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.CONDITIONS.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_CONDITION
        # umap type
        self.UMAP_TYPE = 1
        # Colors 
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_AAT_NOVA

class UMAP1AAT_NOVAPlotConfigCategories(AAT_NOVA_BaseFigureConfig):
    def __init__(self):
        super().__init__()
        # Set the size of the dots
        self.SIZE = 1
        self.ALPHA = 1
        # umap type
        self.UMAP_TYPE = 1
        # How labels are shown in legend
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.CATEGORIES.name
        # Colors 
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_AAT_NOVA_CATEGORIES
        self.COLOR_MAPPINGS_MARKERS = self.COLOR_MAPPINGS_AAT_NOVA_CATEGORIES

class UMAP2ALSPlotConfigAAT_NOVA(AAT_NOVA_BaseFigureConfig):
    def __init__(self):
        super().__init__()

        self.MAP_LABELS_FUNCTION =  MapLabelsFunction.MULTIPLEX_CELL_LINES.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_ALS_AAT_NOVA
        # umap type
        self.UMAP_TYPE = 2

class UMAP0ALSPlotConfigAAT_NOVA(AAT_NOVA_BaseFigureConfig):
    def __init__(self):
        super().__init__()

        self.MAP_LABELS_FUNCTION =  MapLabelsFunction.CELL_LINES.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_ALS_AAT_NOVA
        # umap type
        self.UMAP_TYPE = 0

class UMAP0ALSPlotConfigAAT_NOVA_Cond(AAT_NOVA_BaseFigureConfig):
    def __init__(self):
        super().__init__()
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.CONDITIONS.name
        self.COLOR_MAPPINGS = None
        # umap type
        self.UMAP_TYPE = 0

class UMAP0ALSPlotConfigAAT_NOVAMix(AAT_NOVA_BaseFigureConfig):
    def __init__(self):
        super().__init__()

        self.MAP_LABELS_FUNCTION =  MapLabelsFunction.CELL_LINES.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_ALS_AAT_NOVA
        # umap type
        self.UMAP_TYPE = 0
        self.MIX_GROUPS = True

##########################################################################

class UMAP0PlotConfigAAT_NOVA_by_Cellline(AAT_NOVA_BaseFigureConfig):
    def __init__(self):
        super().__init__()
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.CELL_LINES.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_AAT_NOVA_CELL_LINES
        # umap type
        self.UMAP_TYPE = 0

class UMAP0PlotConfigAAT_NOVA_by_Cellline_Single_Condition(AAT_NOVA_BaseFigureConfig):
    def __init__(self):
        super().__init__()
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.CELL_LINES_CONDITIONS.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_AAT_NOVA_CONDITIONS_PER_CELL_LINE
        # umap type
        self.UMAP_TYPE = 0

class UMAP0PlotConfigAAT_NOVA_by_Condition(AAT_NOVA_BaseFigureConfig):
    def __init__(self):
        super().__init__()
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.CONDITIONS.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_AAT_NOVA_CONDITIONS
        # umap type
        self.UMAP_TYPE = 0

class UMAP0PlotConfigAAT_NOVA_by_Condition_Single_Cell_Line(AAT_NOVA_BaseFigureConfig):
    def __init__(self):
        super().__init__()
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.CELL_LINES_CONDITIONS.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_AAT_NOVA_CONDITIONS_PER_CELL_LINE
        # umap type
        self.UMAP_TYPE = 0

class UMAP0PlotConfigAAT_NOVA_Cellline_Cond(AAT_NOVA_BaseFigureConfig):
    def __init__(self):
        super().__init__()
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.CELL_LINES_CONDITIONS.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_AAT_NOVA_CONDITIONS_PER_CELL_LINE
        # umap type
        self.UMAP_TYPE = 0