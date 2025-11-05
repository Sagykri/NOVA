import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

from manuscript.plot_config import PlotConfig
from manuscript.manuscript_plot_config import *
from src.datasets.label_utils import MapLabelsFunction

class AAT_NOVA_BaseFigureConfig(PlotConfig):
    def __init__(self):
        super().__init__()

        self.COLOR_MAPPINGS_AAT_NOVA_REPS = {
            'rep1': {self.MAPPINGS_ALIAS_KEY: 'Rep1', self.MAPPINGS_COLOR_KEY: '#F04521'},  
            'rep2': {self.MAPPINGS_ALIAS_KEY: 'Rep2', self.MAPPINGS_COLOR_KEY: '#4343FE'},
        }

        self.COLOR_MAPPINGS_AAT_NOVA_BATCHES = {
            'batch1': {self.MAPPINGS_ALIAS_KEY: 'Batch1', self.MAPPINGS_COLOR_KEY: "#409A14"},  
            'batch2': {self.MAPPINGS_ALIAS_KEY: 'Batch2', self.MAPPINGS_COLOR_KEY: "#8215C1"},
        }

        # Markers
        self.COLOR_MAPPINGS_AAT_NOVA = {
            'DAPI': {self.MAPPINGS_ALIAS_KEY: 'Nucleus', self.MAPPINGS_COLOR_KEY: "#7181C7"},
            'Cas3': {self.MAPPINGS_ALIAS_KEY: 'Cas3', self.MAPPINGS_COLOR_KEY: "#3030AC"},
            'FK-2': {self.MAPPINGS_ALIAS_KEY: 'FK-2', self.MAPPINGS_COLOR_KEY: '#921010'},
            'SMI32': {self.MAPPINGS_ALIAS_KEY: 'SMI32', self.MAPPINGS_COLOR_KEY: "#82CD10"},
            'pDRP1': {self.MAPPINGS_ALIAS_KEY: 'pDRP1', self.MAPPINGS_COLOR_KEY: "#1EA072"},
            'TOMM20': {self.MAPPINGS_ALIAS_KEY: 'TOMM20', self.MAPPINGS_COLOR_KEY: "#16AFBA"},
            'pCaMKIIa': {self.MAPPINGS_ALIAS_KEY: 'pCaMKIIa', self.MAPPINGS_COLOR_KEY: "#166FA2"},
            'pTDP-43': {self.MAPPINGS_ALIAS_KEY: 'pTDP-43', self.MAPPINGS_COLOR_KEY: "#8825E5"},
            'TDP-43': {self.MAPPINGS_ALIAS_KEY: 'TDP-43', self.MAPPINGS_COLOR_KEY: "#C620D2"},
            'ATF6': {self.MAPPINGS_ALIAS_KEY: 'ATF6', self.MAPPINGS_COLOR_KEY: "#AC166E"},
            'pAMPK': {self.MAPPINGS_ALIAS_KEY: 'pAMPK', self.MAPPINGS_COLOR_KEY: "#F49DD2"},
            'HDGFL2': {self.MAPPINGS_ALIAS_KEY: 'HDGFL2', self.MAPPINGS_COLOR_KEY: "#FE3B14"},
            'pS6': {self.MAPPINGS_ALIAS_KEY: 'pS6', self.MAPPINGS_COLOR_KEY: "#FBA401"},
            'PAR': {self.MAPPINGS_ALIAS_KEY: 'PAR', self.MAPPINGS_COLOR_KEY: "#FD0B0B"},
            'UNC13A': {self.MAPPINGS_ALIAS_KEY: 'UNC13A', self.MAPPINGS_COLOR_KEY: "#140F4B"},
            'Calreticulin': {self.MAPPINGS_ALIAS_KEY: 'Calreticulin', self.MAPPINGS_COLOR_KEY: 'gray'},
            'LC3-II': {self.MAPPINGS_ALIAS_KEY: 'LC3-II', self.MAPPINGS_COLOR_KEY: "#043121"},
            'p62': {self.MAPPINGS_ALIAS_KEY: 'p62', self.MAPPINGS_COLOR_KEY: "#916706"},
            'CathepsinD': {self.MAPPINGS_ALIAS_KEY: 'CathepsinD', self.MAPPINGS_COLOR_KEY: "#0A4360"},
        }

        # conditions
        self.COLOR_MAPPINGS_AAT_NOVA_CONDITIONS = {
            'PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'PPP2R1A', self.MAPPINGS_COLOR_KEY: "#5D73D6"},
            'HMGCS1': {self.MAPPINGS_ALIAS_KEY: 'HMGCS1', self.MAPPINGS_COLOR_KEY: "#14149A"},
            'PIK3C3': {self.MAPPINGS_ALIAS_KEY: 'PIK3C3', self.MAPPINGS_COLOR_KEY: '#921010'},
            'NDUFAB1': {self.MAPPINGS_ALIAS_KEY: 'NDUFAB1', self.MAPPINGS_COLOR_KEY: "#1EC439"},
            'MAPKAP1': {self.MAPPINGS_ALIAS_KEY: 'MAPKAP1', self.MAPPINGS_COLOR_KEY: "#368664"},
            'NDUFS2': {self.MAPPINGS_ALIAS_KEY: 'NDUFS2', self.MAPPINGS_COLOR_KEY: "#10766B"},
            'RALA': {self.MAPPINGS_ALIAS_KEY: 'RALA', self.MAPPINGS_COLOR_KEY: '#168FB2'},
            'TLK1': {self.MAPPINGS_ALIAS_KEY: 'TLK1', self.MAPPINGS_COLOR_KEY: '#A80358'},
            'NRIP1': {self.MAPPINGS_ALIAS_KEY: 'NRIP1', self.MAPPINGS_COLOR_KEY: "#490092"},
            'TARDBP': {self.MAPPINGS_ALIAS_KEY: 'TARDBP', self.MAPPINGS_COLOR_KEY: "#9223A9"},
            'RANBP17': {self.MAPPINGS_ALIAS_KEY: 'RANBP17', self.MAPPINGS_COLOR_KEY: "#B07FB4"},
            'CYLD': {self.MAPPINGS_ALIAS_KEY: 'CYLD', self.MAPPINGS_COLOR_KEY: '#F04521'},
            'NT-1873': {self.MAPPINGS_ALIAS_KEY: 'NT-1873', self.MAPPINGS_COLOR_KEY: '#F08F21'},
            'NT-6301-3085': {self.MAPPINGS_ALIAS_KEY: 'NT-6301-3085', self.MAPPINGS_COLOR_KEY: "#646464"},
            'Intergenic': {self.MAPPINGS_ALIAS_KEY: 'Intergenic', self.MAPPINGS_COLOR_KEY: "#51388A"},
            'Untreated': {self.MAPPINGS_ALIAS_KEY: 'Untreated', self.MAPPINGS_COLOR_KEY: "#1E1E3A"},
        }

        # Conditions per cell line
        self.COLOR_MAPPINGS_AAT_NOVA_CONDITIONS_PER_CELL_LINE = {
            'CTL_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'CTL_PPP2R1A', self.MAPPINGS_COLOR_KEY: "#5B76D8"}, # brighter
            'C9_PPP2R1A': {self.MAPPINGS_ALIAS_KEY: 'C9_PPP2R1A', self.MAPPINGS_COLOR_KEY: "#1B39A6"}, # darker

            'CTL_HMGCS1': {self.MAPPINGS_ALIAS_KEY: 'CTL_HMGCS1', self.MAPPINGS_COLOR_KEY: "#8989F5"},
            'C9_HMGCS1': {self.MAPPINGS_ALIAS_KEY: 'C9_HMGCS1', self.MAPPINGS_COLOR_KEY: "#2E2E9D"},

            'CTL_PIK3C3': {self.MAPPINGS_ALIAS_KEY: 'CTL_PIK3C3', self.MAPPINGS_COLOR_KEY: "#F34545"},
            'C9_PIK3C3': {self.MAPPINGS_ALIAS_KEY: 'C9_PIK3C3', self.MAPPINGS_COLOR_KEY: "#7D0F0F"},

            'CTL_NDUFAB1': {self.MAPPINGS_ALIAS_KEY: 'CTL_NDUFAB1', self.MAPPINGS_COLOR_KEY: "#33F979"},
            'C9_NDUFAB1': {self.MAPPINGS_ALIAS_KEY: 'C9_NDUFAB1', self.MAPPINGS_COLOR_KEY: "#0B663F"},

            'CTL_MAPKAP1': {self.MAPPINGS_ALIAS_KEY: 'CTL_MAPKAP1', self.MAPPINGS_COLOR_KEY: "#59C3CD"},
            'C9_MAPKAP1': {self.MAPPINGS_ALIAS_KEY: 'C9_MAPKAP1', self.MAPPINGS_COLOR_KEY: "#20678D"},

            'CTL_NDUFS2': {self.MAPPINGS_ALIAS_KEY: 'CTL_NDUFS2', self.MAPPINGS_COLOR_KEY: "#A281E9"},
            'C9_NDUFS2': {self.MAPPINGS_ALIAS_KEY: 'C9_NDUFS2', self.MAPPINGS_COLOR_KEY: "#4D0E99"},

            'CTL_RALA': {self.MAPPINGS_ALIAS_KEY: 'CTL_RALA', self.MAPPINGS_COLOR_KEY: "#A734EF"},
            'C9_RALA': {self.MAPPINGS_ALIAS_KEY: 'C9_RALA', self.MAPPINGS_COLOR_KEY: "#6C1068"},

            'CTL_TLK1': {self.MAPPINGS_ALIAS_KEY: 'CTL_TLK1', self.MAPPINGS_COLOR_KEY: "#E73E8A"},
            'C9_TLK1': {self.MAPPINGS_ALIAS_KEY: 'C9_TLK1', self.MAPPINGS_COLOR_KEY: '#6A0231'},

            'CTL_NRIP1': {self.MAPPINGS_ALIAS_KEY: 'CTL_NRIP1', self.MAPPINGS_COLOR_KEY: "#EB728B"},
            'C9_NRIP1': {self.MAPPINGS_ALIAS_KEY: 'C9_NRIP1', self.MAPPINGS_COLOR_KEY: "#98011F"},

            'CTL_TARDBP': {self.MAPPINGS_ALIAS_KEY: 'CTL_TARDBP', self.MAPPINGS_COLOR_KEY: "#F80000"},
            'C9_TARDBP': {self.MAPPINGS_ALIAS_KEY: 'C9_TARDBP', self.MAPPINGS_COLOR_KEY: "#800606"},

            'CTL_RANBP17': {self.MAPPINGS_ALIAS_KEY: 'CTL_RANBP17', self.MAPPINGS_COLOR_KEY: "#B289B2"},
            'C9_RANBP17': {self.MAPPINGS_ALIAS_KEY: 'C9_RANBP17', self.MAPPINGS_COLOR_KEY: '#914691'},

            'CTL_CYLD': {self.MAPPINGS_ALIAS_KEY: 'CTL_CYLD', self.MAPPINGS_COLOR_KEY: '#FF6F4A'},
            'C9_CYLD': {self.MAPPINGS_ALIAS_KEY: 'C9_CYLD', self.MAPPINGS_COLOR_KEY: '#91260E'},

            'CTL_NT-1873': {self.MAPPINGS_ALIAS_KEY: 'CTL_NT-1873', self.MAPPINGS_COLOR_KEY: "#CA945A"},
            'C9_NT-1873': {self.MAPPINGS_ALIAS_KEY: 'C9_NT-1873', self.MAPPINGS_COLOR_KEY: "#B34902"},

            'CTL_NT-6301-3085': {self.MAPPINGS_ALIAS_KEY: 'CTL_NT-6301-3085', self.MAPPINGS_COLOR_KEY: "#636161"},
            'C9_NT-6301-3085': {self.MAPPINGS_ALIAS_KEY: 'C9_NT-6301-3085', self.MAPPINGS_COLOR_KEY: "#010101"},

            'CTL_Intergenic': {self.MAPPINGS_ALIAS_KEY: 'CTL_Intergenic', self.MAPPINGS_COLOR_KEY: "#B1C543"},
            'C9_Intergenic': {self.MAPPINGS_ALIAS_KEY: 'C9_Intergenic', self.MAPPINGS_COLOR_KEY: "#667D13"},

            'CTL_Untreated': {self.MAPPINGS_ALIAS_KEY: 'CTL_Untreated', self.MAPPINGS_COLOR_KEY: "#D2BF32"},
            'C9_Untreated': {self.MAPPINGS_ALIAS_KEY: 'C9_Untreated', self.MAPPINGS_COLOR_KEY: "#AB8511"},

            'CTL_combined-NT': {self.MAPPINGS_ALIAS_KEY: 'CTL_combined-NT', self.MAPPINGS_COLOR_KEY: "#4B8805"},
            'C9_combined-NT': {self.MAPPINGS_ALIAS_KEY: 'C9_combined-NT', self.MAPPINGS_COLOR_KEY: "#5E11AB"},
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
            'pTDP-43': {self.MAPPINGS_ALIAS_KEY: 'Proteostasis', self.MAPPINGS_COLOR_KEY: '#1F77B4'},
            'TDP-43': {self.MAPPINGS_ALIAS_KEY: 'Proteostasis', self.MAPPINGS_COLOR_KEY: '#1F77B4'},
            'p62': {self.MAPPINGS_ALIAS_KEY: 'Proteostasis', self.MAPPINGS_COLOR_KEY: '#1F77B4'},
            'LC3-II': {self.MAPPINGS_ALIAS_KEY: 'Proteostasis', self.MAPPINGS_COLOR_KEY: '#1F77B4'},
            'FK-2': {self.MAPPINGS_ALIAS_KEY: 'Proteostasis', self.MAPPINGS_COLOR_KEY: '#1F77B4'},
            'pS6': {self.MAPPINGS_ALIAS_KEY: 'Proteostasis', self.MAPPINGS_COLOR_KEY: '#1F77B4'},

            # Splicing ( red)
            'HDGFL2': {self.MAPPINGS_ALIAS_KEY: 'Splicing', self.MAPPINGS_COLOR_KEY: '#E41A1C'},
            'UNC13A': {self.MAPPINGS_ALIAS_KEY: 'Splicing', self.MAPPINGS_COLOR_KEY: '#E41A1C'},

            #Cell Death (green)
            'Cas3': {self.MAPPINGS_ALIAS_KEY: 'Cell Death', self.MAPPINGS_COLOR_KEY: '#4DAF4A'},

            # Metabolism (purple)   
            'pAMPK': {self.MAPPINGS_ALIAS_KEY: 'Metabolism', self.MAPPINGS_COLOR_KEY: '#984EA3'},
            'TOMM20': {self.MAPPINGS_ALIAS_KEY: 'Metabolism', self.MAPPINGS_COLOR_KEY: '#984EA3'},
            'pDRP1': {self.MAPPINGS_ALIAS_KEY: 'Metabolism', self.MAPPINGS_COLOR_KEY: '#984EA3'},

            #Cellular Stress (orange)
            'ATF6': {self.MAPPINGS_ALIAS_KEY: 'Cellular Stress', self.MAPPINGS_COLOR_KEY: '#FF7F00'},
            'Calreticulin': {self.MAPPINGS_ALIAS_KEY: 'Cellular Stress', self.MAPPINGS_COLOR_KEY: '#FF7F00'},
            'CathepsinD': {self.MAPPINGS_ALIAS_KEY: 'Cellular Stress', self.MAPPINGS_COLOR_KEY: '#FF7F00'},
            'PAR': {self.MAPPINGS_ALIAS_KEY: 'Cellular Stress', self.MAPPINGS_COLOR_KEY: '#FF7F00'},
            'SMI32': {self.MAPPINGS_ALIAS_KEY: 'Cellular Stress', self.MAPPINGS_COLOR_KEY: '#FF7F00'},
            'pCaMKIIa': {self.MAPPINGS_ALIAS_KEY: 'Cellular Stress', self.MAPPINGS_COLOR_KEY: '#FF7F00'},
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

class UMAP0PlotConfigAAT_NOVA_by_Reps(AAT_NOVA_BaseFigureConfig):
    def __init__(self):
        super().__init__()
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.REPS.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_AAT_NOVA_REPS
        # umap type
        self.UMAP_TYPE = 0

class UMAP0PlotConfigAAT_NOVA_by_Batches(AAT_NOVA_BaseFigureConfig):
    def __init__(self):
        super().__init__()
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.BATCHES.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_AAT_NOVA_BATCHES
        # umap type
        self.UMAP_TYPE = 0

class UMAP0PlotConfigAAT_NOVA_by_Cellline(AAT_NOVA_BaseFigureConfig):
    def __init__(self):
        super().__init__()
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.CELL_LINES.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_AAT_NOVA_CELL_LINES
        # umap type
        self.UMAP_TYPE = 0

class UMAP0PlotConfigAAT_NOVA_by_Condition(AAT_NOVA_BaseFigureConfig):
    def __init__(self):
        super().__init__()
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.CONDITIONS.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_AAT_NOVA_CONDITIONS
        # umap type
        self.UMAP_TYPE = 0

class UMAP0PlotConfigAAT_NOVA_by_Cellline_and_Condition(AAT_NOVA_BaseFigureConfig):
    def __init__(self):
        super().__init__()
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.CELL_LINES_CONDITIONS.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_AAT_NOVA_CONDITIONS_PER_CELL_LINE
        # umap type
        self.UMAP_TYPE = 0
        # Set the size of the dots
        self.SIZE = 13
        # Set the alpha of the dots (0=max opacity, 1=no opacity)
        self.ALPHA = 0.72


# color by markers
class UMAP1PlotConfigAAT_NOVA(AAT_NOVA_BaseFigureConfig):
    def __init__(self):
        super().__init__()
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.MARKERS.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_AAT_NOVA
        # umap type
        self.UMAP_TYPE = 1

# per marker groups
class UMAP1PlotConfigAAT_NOVA_Per_Marker_Group(AAT_NOVA_BaseFigureConfig):
    def __init__(self):
        super().__init__()
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.MARKERS.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_AAT_NOVA_CATEGORIES
        # umap type
        self.UMAP_TYPE = 1


# color by markers and conditions
class UMAP1PlotConfigAAT_NOVA_Per_Condition(AAT_NOVA_BaseFigureConfig):
        def __init__(self):
            super().__init__()
            self.MAP_LABELS_FUNCTION = MapLabelsFunction.MARKERS.name
            self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_AAT_NOVA_MARKER_PER_CONDITIONS
            # umap type
            self.UMAP_TYPE = 1

class EffectSizePlotConfigAAT_NOVA_Markers(AAT_NOVA_BaseFigureConfig):
    def __init__(self):
        super().__init__()
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_AAT_NOVA
        self.COLOR_MAPPINGS_MARKERS = self.COLOR_MAPPINGS_AAT_NOVA
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.MARKERS.name
        self.COLOR_MAPPINGS_CELL_LINE_CONDITION = self.COLOR_MAPPINGS_AAT_NOVA_CONDITIONS_PER_CELL_LINE
        self.FIGSIZE = (5,7)

class EffectSizePlotConfigAAT_NOVA_CellLines_Conditions(AAT_NOVA_BaseFigureConfig):
    def __init__(self):
        super().__init__()
        self.MAP_LABELS_FUNCTION = MapLabelsFunction.CELL_LINES_CONDITIONS.name
        self.COLOR_MAPPINGS = self.COLOR_MAPPINGS_AAT_NOVA_CONDITIONS_PER_CELL_LINE
        self.FIGSIZE = (5,7)