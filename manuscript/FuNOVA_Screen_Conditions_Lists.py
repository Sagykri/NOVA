import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))

############### PLATE1 ############### 
plate1_conditions = ['CTNNB1', 'CTNNBL1', 'CYP46A1', 'DDHD1', 'DLD', 'ETV5', 'FAM149B1', 'FAM171A1', 'FBXO10', 'FGD1', 'FNBP1L', 'FOXK1', 'GABRA2', 'GAK', 'GHITM', 'GNB1L', 'GNG4', 'GRAMD4', 'HDAC2', 'BANP', 'HNRNPU', 'INPP5F', 'INTS8', 'IQSEC2', 'KIF3B', 'KIFAP3', 'LDLR', 'LRIF1', 'LRP1B', 'BOLA3', 'MAP3K4', 'MAP4', 'MBTPS1', 'MED15', 'MXRA8', 'NAE1', 'NAV1', 'NEIL1', 'BRD9', 'NFASC', 'NRP1', 'NUDCD3', 'non-targeting-00004-00017-p1', 'non-targeting-00010-00031-p1', 'non-targeting-00035-00050-p1', 'non-targeting-00053-00059-p1', 'non-targeting-00111-00121-p1', 'Empty-p1']

plate1_control_conditions = ['non-targeting-00004-00017-p1', 'non-targeting-00010-00031-p1', 'non-targeting-00035-00050-p1', 'non-targeting-00053-00059-p1', 'non-targeting-00111-00121-p1', 'Empty-p1']

plate1_kds = ['CTNNB1', 'CTNNBL1', 'CYP46A1', 'DDHD1', 'DLD', 'ETV5', 'FAM149B1', 'FAM171A1', 'FBXO10', 'FGD1', 'FNBP1L', 'FOXK1', 'GABRA2', 'GAK', 'GHITM', 'GNB1L', 'GNG4', 'GRAMD4', 'HDAC2', 'BANP', 'HNRNPU', 'INPP5F', 'INTS8', 'IQSEC2', 'KIF3B', 'KIFAP3', 'LDLR', 'LRIF1', 'LRP1B', 'BOLA3', 'MAP3K4', 'MAP4', 'MBTPS1', 'MED15', 'MXRA8', 'NAE1', 'NAV1', 'NEIL1', 'BRD9', 'NFASC', 'NRP1', 'NUDCD3']

############### PLATE2 ############### 
plate2_conditions= ['NUP133', 'OGFOD1', 'PAK1', 'PAQR7', 'PDLIM4', 'PDZRN3', 'PHF2', 'CASD1', 'POLR3E', 'POMGNT2', 'PRIM1', 'RBBP6', 'RECK', 'RNF20', 'RUSC2', 'SEH1L', 'SENP5', 'CC2D2A', 'SMC1A', 'SNX30', 'STMN2', 'SYT16', 'TAF1', 'TBCD', 'TBCE', 'TELO2', 'TIMM21', 'TMEM50B', 'TMOD1', 'TMX2', 'TUBA1A', 'UROS', 'COPS8', 'ZBTB18', 'ZNF462', 'CTNNA2', 'CCDC152', 'CCDC183', 'CDH20', 'CDK7', 'CEP57L1', 'CHMP7', 'non-targeting-00004-00017-p2', 'non-targeting-00010-00031-p2', 'non-targeting-00035-00050-p2', 'non-targeting-00053-00059-p2', 'non-targeting-00111-00121-p2', 'Empty-p2']

plate2_control_conditions = ['non-targeting-00004-00017-p2', 'non-targeting-00010-00031-p2', 'non-targeting-00035-00050-p2', 'non-targeting-00053-00059-p2', 'non-targeting-00111-00121-p2', 'Empty-p2']

plate2_kds = ['NUP133', 'OGFOD1', 'PAK1', 'PAQR7', 'PDLIM4', 'PDZRN3', 'PHF2', 'CASD1', 'POLR3E', 'POMGNT2', 'PRIM1', 'RBBP6', 'RECK', 'RNF20', 'RUSC2', 'SEH1L', 'SENP5', 'CC2D2A', 'SMC1A', 'SNX30', 'STMN2', 'SYT16', 'TAF1', 'TBCD', 'TBCE', 'TELO2', 'TIMM21', 'TMEM50B', 'TMOD1', 'TMX2', 'TUBA1A', 'UROS', 'COPS8', 'ZBTB18', 'ZNF462', 'CTNNA2', 'CCDC152', 'CCDC183', 'CDH20', 'CDK7', 'CEP57L1', 'CHMP7']

############### PLATE3 ############### 

plate3_conditions = ['CSNK1E', 'DCX', 'ANXA11', 'DHX37', 'DLG5', 'DLGAP1', 'DLGAP2', 'DOC2A', 'DPY19L4', 'EFCAB2', 'GADD45A', 'ASRGL1', 'GNAL', 'GPCPD1', 'GRM2', 'MCM7', 'ATF7IP', 'MNAT1', 'MTMR3', 'MYCN', 'NADSYN1', 'NCDN', 'NECAB3', 'NME3', 'NOMO2', 'OSBP2', 'POLI', 'TDP-43-p3', 'Ranbp17-p3', 'PSD', 'RHBDD1', 'PTGIS', 'RPL29', 'RYR3', 'SEMA6C', 'SGF29', 'SLC39A9', 'SRR', 'BSN', 'SYN3', 'TCF4', 'TENM1', 'non-targeting-00004-00017-p3', 'non-targeting-00010-00031-p3', 'non-targeting-00035-00050-p3', 'non-targeting-00053-00059-p3', 'non-targeting-00111-00121-p3', 'Empty-p3']

plate3_control_conditions = ['non-targeting-00004-00017-p3', 'non-targeting-00010-00031-p3', 'non-targeting-00035-00050-p3', 'non-targeting-00053-00059-p3', 'non-targeting-00111-00121-p3', 'Empty-p3']

plate3_kds = ['CSNK1E', 'DCX', 'ANXA11', 'DHX37', 'DLG5', 'DLGAP1', 'DLGAP2', 'DOC2A', 'DPY19L4', 'EFCAB2', 'GADD45A', 'ASRGL1', 'GNAL', 'GPCPD1', 'GRM2', 'MCM7', 'ATF7IP', 'MNAT1', 'MTMR3', 'MYCN', 'NADSYN1', 'NCDN', 'NECAB3', 'NME3', 'NOMO2', 'OSBP2', 'POLI', 'TDP-43-p3', 'Ranbp17-p3', 'PSD', 'RHBDD1', 'PTGIS', 'RPL29', 'RYR3', 'SEMA6C', 'SGF29', 'SLC39A9', 'SRR', 'BSN', 'SYN3', 'TCF4', 'TENM1']

############### PLATE4 ###############
plate4_conditions = ['TFB1M', 'THSD7A', 'THY1', 'TMEM30A', 'WDR86', 'ZNF566', 'ZNF652', 'CAPRIN2', 'CCDC146', 'AKT3', 'ZNF772', 'AKIRIN2', 'CDK12', 'FAM193B', 'INVS', 'MAPKAP1', 'MYO1C', 'POLD1', 'RCOR2', 'SOBP', 'TDP1', 'TRRAP', 'USP24', 'ATF7IP2', 'BMP2K', 'CACNA2D2', 'CHMP2B', 'CYP51A1', 'ERO1B', 'HIVEP2', 'HSP90AB1', 'HSPH1', 'INPP5B', 'KDM1B', 'KIAA0232', 'NECAB2', 'ONECUT2', 'PIGZ', 'HMGCS1', 'PIK3C3', 'Ranbp17-p4', 'TDP-43-p4', 'non-targeting-00004-00017-p4', 'non-targeting-00010-00031-p4', 'non-targeting-00035-00050-p4', 'non-targeting-00053-00059-p4', 'non-targeting-00111-00121-p4', 'Empty-p4']

plate4_control_conditions = ['non-targeting-00004-00017-p4', 'non-targeting-00010-00031-p4', 'non-targeting-00035-00050-p4', 'non-targeting-00053-00059-p4', 'non-targeting-00111-00121-p4', 'Empty-p4']

plate4_kds = ['TFB1M', 'THSD7A', 'THY1', 'TMEM30A', 'WDR86', 'ZNF566', 'ZNF652', 'CAPRIN2', 'CCDC146', 'AKT3', 'ZNF772', 'AKIRIN2', 'CDK12', 'FAM193B', 'INVS', 'MAPKAP1', 'MYO1C', 'POLD1', 'RCOR2', 'SOBP', 'TDP1', 'TRRAP', 'USP24', 'ATF7IP2', 'BMP2K', 'CACNA2D2', 'CHMP2B', 'CYP51A1', 'ERO1B', 'HIVEP2', 'HSP90AB1', 'HSPH1', 'INPP5B', 'KDM1B', 'KIAA0232', 'NECAB2', 'ONECUT2', 'PIGZ', 'HMGCS1', 'PIK3C3', 'Ranbp17-p4', 'TDP-43-p4']