import os
import sys
import pandas as pd
sys.path.insert(1, os.getenv("NOVA_HOME"))

PATH = "/home/projects/hornsteinlab/Collaboration/Guy_Lior/fuNOVA_Screen_B2/genes_metadata_with_names.csv"

df = pd.read_csv(PATH)

def get_conditions_by_plate(plate_num):
    plate_conditions = df[df['plate'] == plate_num]['gene_name'].tolist()
    control_conditions = [condition for condition in plate_conditions if ("non-targeting" in condition) or ("Empty" in condition)]

    kds = list(set(plate_conditions) - set(control_conditions))

    return plate_conditions, control_conditions, kds

############### PLATE1 ############### 
# extracted by -plate1_conditions, plate1_control_conditions, plate1_kds = get_conditions_by_plate(1)

plate1_conditions =  ['DDHD1', 'CYP46A1', 'CTNNBL1', 'CTNNB1', 'FAM171A1', 'FAM149B1', 'ETV5', 'DLD', 'FOXK1', 'FNBP1L', 'FGD1', 'FBXO10', 'GNB1L', 'GHITM', 'GAK', 'GABRA2', 'BANP','HDAC2', 'GRAMD4', 'GNG4', 'IQSEC2', 'INTS8', 'INPP5F', 'HNRNPU', 'LRIF1', 'LDLR', 'KIFAP3', 'KIF3B', 'MAP4', 'MAP3K4', 'BOLA3', 'LRP1B', 'NAE1', 'MXRA8', 'MED15', 'MBTPS1', 'NFASC', 'BRD9', 'NEIL1', 'NAV1', 'non-targeting-00010-00031-p1', 'non-targeting-00004-00017-p1', 'NUDCD3', 'NRP1', 'non-targeting-00111-00121-p1', 'non-targeting-00053-00059-p1', 'non-targeting-00035-00050-p1', 'Empty-p1']
plate1_control_conditions = ['non-targeting-00010-00031-p1', 'non-targeting-00004-00017-p1', 'non-targeting-00111-00121-p1', 'non-targeting-00053-00059-p1', 'non-targeting-00035-00050-p1', 'Empty-p1']
plate1_kds =  ['ETV5', 'FNBP1L', 'INTS8', 'NEIL1', 'GHITM', 'HNRNPU', 'MXRA8', 'MED15', 'CTNNB1', 'CTNNBL1', 'MAP3K4', 'GNB1L', 'GAK', 'GNG4', 'HDAC2', 'DDHD1', 'LDLR', 'GRAMD4', 'DLD', 'KIF3B', 'FAM149B1', 'IQSEC2', 'BOLA3', 'LRP1B', 'LRIF1', 'NAE1', 'FGD1', 'FBXO10', 'CYP46A1', 'BANP', 'BRD9', 'KIFAP3', 'NAV1', 'INPP5F', 'MAP4', 'FAM171A1', 'MBTPS1', 'NUDCD3', 'NRP1', 'FOXK1', 'GABRA2', 'NFASC']


############### PLATE2 ############### 
# plate2_conditions, plate2_control_conditions, plate2_kds = get_conditions_by_plate(2)

plate2_conditions =  ['PAQR7', 'PAK1', 'OGFOD1', 'NUP133', 'CASD1', 'PHF2', 'PDZRN3', 'PDLIM4', 'RBBP6', 'PRIM1', 'POMGNT2', 'POLR3E', 'SEH1L', 'RUSC2', 'RNF20', 'RECK', 'SNX30', 'SMC1A', 'CC2D2A', 'SENP5', 'TBCD', 'TAF1', 'SYT16', 'STMN2', 'TMEM50B', 'TIMM21', 'TELO2', 'TBCE', 'UROS', 'TUBA1A', 'TMX2', 'TMOD1', 'CTNNA2', 'ZNF462', 'ZBTB18', 'COPS8', 'CDK7', 'CDH20', 'CCDC183', 'CCDC152', 'non-targeting-00010-00031-p2', 'non-targeting-00004-00017-p2', 'CHMP7', 'CEP57L1', 'non-targeting-00111-00121-p2', 'non-targeting-00053-00059-p2', 'non-targeting-00035-00050-p2', 'Empty-p2']
plate2_control_conditions = ['non-targeting-00010-00031-p2', 'non-targeting-00004-00017-p2', 'non-targeting-00111-00121-p2', 'non-targeting-00053-00059-p2', 'non-targeting-00035-00050-p2', 'Empty-p2']
plate2_kds =  ['RNF20', 'SEH1L', 'TELO2', 'NUP133', 'CCDC183', 'PRIM1', 'TBCD', 'TBCE', 'COPS8', 'SYT16', 'OGFOD1', 'RECK', 'UROS', 'CDK7', 'ZNF462', 'TIMM21', 'SNX30', 'RBBP6', 'CTNNA2', 'CEP57L1', 'POLR3E', 'POMGNT2', 'ZBTB18', 'TMOD1', 'TMEM50B', 'PDLIM4', 'PAK1', 'PHF2', 'STMN2', 'CASD1', 'TAF1', 'CHMP7', 'PDZRN3', 'CC2D2A', 'CCDC152', 'PAQR7', 'TMX2', 'CDH20', 'SENP5', 'SMC1A', 'TUBA1A', 'RUSC2']

############### PLATE3 ############### 
# plate3_conditions, plate3_control_conditions, plate3_kds = get_conditions_by_plate(3)
plate3_conditions =  ['DHX37', 'ANXA11', 'DCX', 'CSNK1E', 'DOC2A', 'DLGAP2', 'DLGAP1', 'DLG5', 'ASRGL1', 'GADD45A', 'EFCAB2', 'DPY19L4', 'MCM7', 'GRM2', 'GPCPD1', 'GNAL', 'MYCN', 'MTMR3', 'MNAT1', 'ATF7IP', 'NME3', 'NECAB3', 'NCDN', 'NADSYN1', 'TDP-43-p3', 'POLI', 'OSBP2', 'NOMO2', 'PTGIS', 'RHBDD1', 'PSD', 'Ranbp17-p3', 'SGF29', 'SEMA6C', 'RYR3', 'RPL29', 'SYN3', 'BSN', 'SRR', 'SLC39A9', 'non-targeting-00010-00031-p3', 'non-targeting-00004-00017-p3', 'TENM1', 'TCF4', 'non-targeting-00111-00121-p3', 'non-targeting-00053-00059-p3', 'non-targeting-00035-00050-p3', 'Empty-p3']
plate3_control_conditions = ['non-targeting-00010-00031-p3', 'non-targeting-00004-00017-p3', 'non-targeting-00111-00121-p3', 'non-targeting-00053-00059-p3', 'non-targeting-00035-00050-p3', 'Empty-p3']
plate3_kds =  ['DLGAP2', 'ASRGL1', 'NADSYN1', 'SYN3', 'RYR3', 'EFCAB2', 'Ranbp17-p3', 'CSNK1E', 'DCX', 'MNAT1', 'GPCPD1', 'ATF7IP', 'NCDN', 'SEMA6C', 'DHX37', 'PSD', 'SRR', 'DLGAP1', 'NOMO2', 'BSN', 'SLC39A9', 'MTMR3', 'MCM7', 'DOC2A', 'GRM2', 'OSBP2', 'RHBDD1', 'NME3', 'DLG5', 'POLI', 'TENM1', 'GNAL', 'DPY19L4', 'MYCN', 'SGF29', 'PTGIS', 'TDP-43-p3', 'TCF4', 'NECAB3', 'RPL29', 'ANXA11', 'GADD45A']

############### PLATE4 ###############
# plate4_conditions, plate4_control_conditions, plate4_kds = get_conditions_by_plate(4)

plate4_conditions =  ['TMEM30A', 'THY1', 'THSD7A', 'TFB1M', 'CAPRIN2', 'ZNF652', 'ZNF566', 'WDR86', 'AKIRIN2', 'ZNF772', 'AKT3', 'CCDC146', 'MAPKAP1', 'INVS', 'FAM193B', 'CDK12', 'SOBP', 'RCOR2', 'POLD1', 'PDCD10', 'ATF7IP2', 'USP24', 'TRRAP', 'TDP1', 'CYP51A1', 'CHMP2B', 'CACNA2D2', 'BMP2K', 'HSPH1', 'HSP90AB1', 'HIVEP2', 'ERO1B', 'NECAB2', 'KIAA0232', 'KDM1B', 'INPP5B', 'PIK3C3', 'HMGCS1', 'PIGZ', 'ONECUT2', 'non-targeting-00010-00031-p4', 'non-targeting-00004-00017-p4', 'TDP-43-p4', 'Ranbp17-p4', 'non-targeting-00111-00121-p4', 'non-targeting-00053-00059-p4', 'non-targeting-00035-00050-p4', 'Empty-p4']
plate4_control_conditions = ['non-targeting-00010-00031-p4', 'non-targeting-00004-00017-p4', 'non-targeting-00111-00121-p4', 'non-targeting-00053-00059-p4', 'non-targeting-00035-00050-p4', 'Empty-p4']
plate4_kds =  ['HSP90AB1', 'THY1', 'ONECUT2', 'USP24', 'TRRAP', 'CYP51A1', 'INVS', 'PIGZ', 'CHMP2B', 'ERO1B', 'WDR86', 'AKT3', 'SOBP', 'CDK12', 'KIAA0232', 'TDP-43-p4', 'AKIRIN2', 'ZNF772', 'FAM193B', 'NECAB2', 'HSPH1', 'CAPRIN2', 'ZNF566', 'HIVEP2', 'POLD1', 'PDCD10', 'TFB1M', 'PIK3C3', 'TDP1', 'MAPKAP1', 'KDM1B', 'HMGCS1', 'TMEM30A', 'THSD7A', 'CCDC146', 'CACNA2D2', 'INPP5B', 'RCOR2', 'ZNF652', 'ATF7IP2', 'Ranbp17-p4', 'BMP2K']

if __name__ == "__main__":
    print("plate1_conditions = ", plate1_conditions)
    print("plate1_control_conditions =", plate1_control_conditions)
    print("plate1_kds = ", plate1_kds)

    print("\nplate2_conditions = ", plate2_conditions)
    print("plate2_control_conditions =", plate2_control_conditions)
    print("plate2_kds = ", plate2_kds)

    print("\nplate3_conditions = ", plate3_conditions)
    print("plate3_control_conditions =", plate3_control_conditions)
    print("plate3_kds = ", plate3_kds)

    print("\nplate4_conditions = ", plate4_conditions)
    print("plate4_control_conditions =", plate4_control_conditions)
    print("plate4_kds = ", plate4_kds)