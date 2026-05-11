import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))
from src.preprocessing.preprocessing_config import PreprocessingConfig
from manuscript.FuNOVA_Screen_Conditions_Lists import plate1_conditions, plate2_conditions, plate3_conditions, plate4_conditions
FuNOVA_DATA_DIR = "/home/projects/hornsteinlab/Collaboration/Guy_Lior/fuNOVA_Screen"
class PreprocessingBaseConfigFuNOVAScreen(PreprocessingConfig):
    def __init__(self):
        super().__init__()
        self.RAW_FOLDER_ROOT = FuNOVA_DATA_DIR
        self.RAW_FOLDER_ROOT = os.path.join(self.RAW_FOLDER_ROOT,'sorted')
        self.PROCESSED_FOLDER_ROOT = os.path.join(self.PROCESSED_FOLDER_ROOT,'FuNOVA_Screen')
        self.OUTPUTS_FOLDER =  os.path.join(os.getenv("NOVA_HOME"),'outputs','preprocessing','FuNOVA_Screen')
        self.LOGS_FOLDER = os.path.join(self.OUTPUTS_FOLDER,'logs')
        self.PREPROCESSOR_CLASS_PATH = os.path.join("src","preprocessing","preprocessors","preprocessor_opera","OperaPreprocessor")        

        self.CELL_LINES = ["C9"]
        self.CONDITIONS = plate1_conditions + plate2_conditions + plate3_conditions + plate4_conditions
        # ["non-targeting_00004_00017_p1", "non-targeting_00004_00017_p2",
                    #  "non-targeting_00004_00017_p3","non-targeting_00004_00017_p4", 
                    #  "non-targeting_00010_00031_p1", "non-targeting_00010_00031_p2", "non-targeting_00010_00031_p3", "non-targeting_00010_00031_p4","non-targeting_00035_00050_p1", "non-targeting_00035_00050_p2", "non-targeting_00035_00050_p3", "non-targeting_00035_00050_p4", "non-targeting_00053_00059_p1", "non-targeting_00053_00059_p2", "non-targeting_00053_00059_p3", "non-targeting_00053_00059_p4", "non-targeting_00111_00121_p1", "non-targeting_00111_00121_p2", "non-targeting_00111_00121_p3", "non-targeting_00111_00121_p4", 
                    #  "Empty_p1", "Empty_p2", "Empty_p3", "Empty_p4",]
                    #  "TDP-43_p3", "TDP-43_p4", 
                    #  "Ranbp17_p3", "Ranbp17_p4",
                    #  "AKIRIN2","CDH20","DLGAP1","GHITM","KDM1B","NADSYN1","OGFOD1","RBBP6","SYN3","TRRAP","AKT3","CDK12","DLGAP2","GNAL","KIAA0232","NAE1","ONECUT2","RCOR2","SYT16","TUBA1A","ANXA11","CDK7","DOC2A","GNB1L","KIF3B","NAV1","OSBP2","RECK","TAF1","UROS","ASRGL1","CEP57L1","DPY19L4","GNG4","KIFAP3","NCDN","PAK1","RHBDD1","TBCD","USP24","ATF7IP","CHMP2B","EFCAB2","GPCPD1","LDLR","NECAB2","PAQR7","RNF20","TBCE","WDR86","ATF7IP2","CHMP7","GRAMD4","LRIF1","NECAB3","PDLIM4","RPL29","TCF4","ZBTB18","BANP","COPS8","ERO1B","GRM2","LRP1B","NEIL1","PDZRN3","RUSC2","TDP1","ZNF462","BMP2K","CSNK1E","ETV5","HDAC2","MAP3K4","NFASC","PHF2","RYR3","ZNF566","BOLA3","CTNNA2","FAM149B1","HIVEP2","MAP4","NME3","PIGZ","SEH1L","TELO2","ZNF652","BRD9","CTNNB1","FAM171A1","HMGCS1","MAPKAP1","NOMO2","PIK3C3","SEMA6C","TENM1","ZNF772","BSN","CTNNBL1","FAM193B","HNRNPU","MBTPS1","POLD1","SENP5","TFB1M","CACNA2D2","CYP46A1","FBXO10","HSP90AB1","MCM7","POLI","SGF29","THSD7A","CAPRIN2","CYP51A1","FGD1","HSPH1","MED15","POLR3E","SLC39A9","THY1","CASD1","DCX","FNBP1L","INPP5B","MNAT1","POMGNT2","SMC1A","TIMM21","CC2D2A","DDHD1","FOXK1","INPP5F","MTMR3","PRIM1","SNX30","TMEM30A","CCDC146","DHX37","GABRA2","INTS8","MXRA8","NRP1","PSD","SOBP","TMEM50B","CCDC152","DLD","GADD45A","INVS","MYCN","NUDCD3","PTGIS","SRR","TMOD1","CCDC183","DLG5","GAK","IQSEC2","MYO1C","NUP133","STMN2","TMX2"]
        self.REPS = ["rep1","rep2"]
        self.PANELS = ["Panel1","Panel2","Panel3","Panel4"]
        self.MARKERS = ['DAPI', 'TDP-43', 'p62', 'pTDP-43', 'ATF6', 'pAMPK', 'G3BP1', 'Calreticulin', 'Aggreagtes', 'Cas3', 'pS6'] # DISCARD - 'HDGFL2', 'FK-2'

        self.MARKERS_FOCUS_BOUNDRIES_PATH = os.path.join(
            os.getenv("NOVA_HOME"),
            'manuscript',
            'markers_focus_boundries',
            'markers_focus_boundries_FuNOVA_Screen.csv'
        )


class PreprocessingBaseConfigFuNOVAScreenBatch1(PreprocessingBaseConfigFuNOVAScreen):
    def __init__(self):
        super().__init__()
        
        self.INPUT_FOLDERS = [os.path.join(self.RAW_FOLDER_ROOT,"batch1")]
        self.PROCESSED_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT,"batch1")]
        self.OUTPUTS_FOLDER  = os.path.join(self.OUTPUTS_FOLDER,"batch1")

