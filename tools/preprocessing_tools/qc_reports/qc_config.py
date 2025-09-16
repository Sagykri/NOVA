import pandas as pd
import seaborn as sns
import numpy as np
import natsort
import matplotlib.colors as mcolors
from collections import defaultdict
# TODO: Pretify this


# ----------------------------------------- #
# U2OS
# ----------------------------------------- #

U2OS_panels = pd.DataFrame([
    ['DCP1A'],
    ['G3BP1'],
    ['Phalloidin'],
    ['DAPI']], 
    columns=['A'],
    index=['Cy5', 'mCherry', 'GFP','DAPI'])

U2OS_markers = ['DCP1A','G3BP1','Phalloidin','DAPI']

U2OS_marker_info = pd.DataFrame([[['Cy5']]*10 + [['mCherry']]*11 + [['GFP']]*4,
                          [['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['K'],
                          ['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['J'],['K'],
                          ['A'],['B'],['C'],['K']]], index=['Antibody','panel'],
                         columns = ['G3BP1','NONO','SQSTM1','PSD95','NEMO','GM130',
                                    'NCL','ANXA11','Calreticulin','mitotracker',
                                    'KIF5A','TDP43','FMRP','CLTC','DCP1A','TOMM20',
                                    'FUS','SCNA','LAMP1','TIA1','PML', 'PURA','CD41',
                                    'Phalloidin', 'PEX14']).T  # order here is important!

U2OS_cell_lines = ['WT']

U2OS_cell_lines_to_cond = {'WT':['Untreated','stress']}

U2OS_cell_lines_for_disp = {'WT_stress':'WT_stress', 'WT_Untreated':'WT_Untreated'}

U2OS_reps = ['rep1']

U2OS_colorblind_palette = sns.color_palette('colorblind')

U2OS_line_colors = {
    'WT Untreated': U2OS_colorblind_palette[9],
    'WT stress': U2OS_colorblind_palette[3]
}

U2OS_lines_order = U2OS_line_colors.keys()
U2OS_custom_palette = [U2OS_line_colors[line] for line in U2OS_lines_order]

U2OS_expected_dapi_raw = 144

# ----------------------------------------- #
# iNDi neurons - old (training data)
# ----------------------------------------- #

panels = pd.DataFrame([
    ['G3BP1','NONO','SQSTM1','PSD95','NEMO','GM130','NCL','ANXA11','Calreticulin',np.nan,'mitotracker'],
    ['KIF5A','TDP43','FMRP','CLTC','DCP1A','TOMM20','FUS','SCNA','LAMP1','TIA1','PML'],
    ['PURA','CD41','Phalloidin',np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,'PEX14'],
    ['DAPI']*11], 
    columns=['A','B','C','D','E','F','G','H','I','J','K'],
    index=['Cy5', 'mCherry', 'GFP','DAPI'])

markers = ['G3BP1','NONO','SQSTM1','PSD95','NEMO','GM130',
           'NCL','ANXA11','Calreticulin','mitotracker',
           'KIF5A','TDP43','FMRP','CLTC','DCP1A','TOMM20',
           'FUS','SCNA','LAMP1','TIA1','PML',
           'PURA','CD41','Phalloidin', 'PEX14','DAPI']

marker_info = pd.DataFrame([[['Cy5']]*10 + [['mCherry']]*11 + [['GFP']]*4,
                          [['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['K'],
                          ['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['J'],['K'],
                          ['A'],['B'],['C'],['K']]], index=['Antibody','panel'],
                         columns = ['G3BP1','NONO','SQSTM1','PSD95','NEMO','GM130',
                                    'NCL','ANXA11','Calreticulin','mitotracker',
                                    'KIF5A','TDP43','FMRP','CLTC','DCP1A','TOMM20',
                                    'FUS','SCNA','LAMP1','TIA1','PML', 'PURA','CD41',
                                    'Phalloidin', 'PEX14']).T  # order here is important!

cell_lines = ['FUSHomozygous', 'TDP43', 'TBK1', 'WT', 'FUSRevertant','OPTN', 'FUSHeterozygous', 'SCNA']

cell_lines_to_cond = {'FUSHomozygous':['Untreated'], 'TDP43':['Untreated'], 'TBK1':['Untreated'],
                      'WT':['Untreated','stress'], 'FUSRevertant':['Untreated'],
                      'OPTN':['Untreated'], 'FUSHeterozygous':['Untreated'],'SCNA':['Untreated']}

cell_lines_for_disp = {'FUSHomozygous_Untreated':'FUSHomozygous', 
                       'TDP43_Untreated':'TDP43', 
                       'TBK1_Untreated':'TBK1', 
                       'WT_stress':'WT_stress', 'WT_Untreated':'WT_Untreated',
                       'FUSRevertant_Untreated':'FUSRevertant',
                       'OPTN_Untreated':'OPTN', 
                       'FUSHeterozygous_Untreated':'FUSHeterozygous',
                       'SCNA_Untreated':'SNCA'}

reps = ['rep1','rep2']

colorblind_palette = sns.color_palette('colorblind')

line_colors = {
    'FUSHeterozygous': colorblind_palette[0],
    'FUSHomozygous': colorblind_palette[1],
    'FUSRevertant': colorblind_palette[2],
    'OPTN': colorblind_palette[8],
    'SNCA': colorblind_palette[4],
    'TBK1': colorblind_palette[5],
    'TDP43': colorblind_palette[6],
    'WT Untreated': colorblind_palette[9],
    'WT stress': colorblind_palette[3]
}

lines_order = line_colors.keys()
custom_palette = [line_colors[line] for line in lines_order]

expected_dapi_raw = 1100


# ----------------------------------------- #
# iNDI Day 8 Neurons OPERA (New)
# ----------------------------------------- #
new_d8_panels = pd.DataFrame([
             ['G3BP1','NONO','SQSTM1','PSD95','NEMO','GM130','NCL','LSM14A', 'TDP43', 'ANXA11', 'PEX14', 'mitotracker'],
             ['FMRP','SON','KIF5A', 'CLTC', 'DCP1A', 'Calreticulin', 'FUS', 'HNRNPA1', 'PML', 'LAMP1', 'SNCA', 'TIA1'],
             ['PURA',np.nan,'Tubulin', 'Phalloidin',np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,'TOMM20'],
             ['DAPI']*12], columns=['A','B','C','D','E','F','G','H','I','J','K','L'],
            index=['Cy5', 'mCherry', 'GFP','DAPI'])

new_d8_markers = ['G3BP1','NONO','SQSTM1','PSD95','NEMO','GM130','NCL','LSM14A', 'TDP43', 'ANXA11', 'PEX14', 'mitotracker',
                                 'FMRP','SON','KIF5A', 'CLTC', 'DCP1A', 'Calreticulin', 'FUS', 'HNRNPA1', 'PML', 'LAMP1', 'SNCA', 'TIA1',
                                 'PURA','CD41','Tubulin', 'Phalloidin', 'TOMM20', 'DAPI']

new_d8_marker_info = pd.DataFrame([[['Cy5']]*12 + [['mCherry']]*12 + [['GFP']]*4,
                          [['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['J'],['K'],['L'],
                          ['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['J'],['K'],['L'],
                          ['A'],['C'],['D'],['L']]], index=['Antibody','panel'],
                         columns = ['G3BP1','NONO','SQSTM1','PSD95','NEMO','GM130','NCL', 
                                    'LSM14A', 'TDP43', 'ANXA11', 'PEX14', 'mitotracker',
                                    'FMRP','SON','KIF5A', 'CLTC', 'DCP1A', 'Calreticulin', 
                                    'FUS', 'HNRNPA1', 'PML', 'LAMP1', 'SNCA', 'TIA1',
                                    'PURA','Tubulin', 'Phalloidin', 'TOMM20']).T  # order here is important!

new_d8_cell_lines = ['FUSHomozygous', 'TDP43', 'TBK1', 'WT', 'FUSRevertant','OPTN', 'FUSHeterozygous', 'SCNA']

new_d8_cell_lines_to_cond = {'FUSHomozygous':['Untreated'], 
                             'TDP43':['Untreated'], 
                             'TBK1':['Untreated'],
                             'WT':['Untreated'], 
                             'FUSRevertant':['Untreated'],
                             'OPTN':['Untreated'], 
                             'FUSHeterozygous':['Untreated'],
                             'SNCA':['Untreated']}

new_d8_cell_lines_for_disp = {'FUSHomozygous_Untreated':'FUSHomozygous', 
                              'TDP43_Untreated':'TDP43', 
                              'TBK1_Untreated':'TBK1', 
                              'WT_Untreated':'WT Untreated',
                              'FUSRevertant_Untreated':'FUSRevertant',
                              'OPTN_Untreated':'OPTN', 
                              'FUSHeterozygous_Untreated':'FUSHeterozygous',
                              'SNCA_Untreated':'SNCA'}

new_d8_reps = ['rep1','rep2']

new_d8_colorblind_palette = sns.color_palette('colorblind')

new_d8_line_colors = {
    'FUSHeterozygous': colorblind_palette[0],
    'FUSHomozygous': colorblind_palette[1],
    'FUSRevertant': colorblind_palette[2],
    'OPTN': colorblind_palette[8],
    'SNCA': colorblind_palette[4],
    'TBK1': colorblind_palette[5],
    'TDP43': colorblind_palette[6],
    'WT Untreated': colorblind_palette[9]
}
new_d8_lines_order = new_d8_line_colors.keys()
new_d8_custom_palette = [line_colors[line] for line in lines_order] + [colorblind_palette[7]]
new_d8_expected_dapi_raw = 250*12


# ----------------------------------------- #
# dNLS OPERA (New)
# ----------------------------------------- #
dnls_opera_cell_lines = ['dNLS']

dnls_opera_cell_lines_to_cond = {'dNLS':['DOX','Untreated']}

dnls_opera_panels = pd.DataFrame([
    ['G3BP1','NONO','SQSTM1','PSD95','NEMO','GM130','NCL','LSM14A','TDP43','ANXA11','PEX14','mitotracker'],
    ['FMRP','SON','KIF5A','CLTC','DCP1A','Calreticulin','FUS','HNRNPA1','PML','LAMP1','SNCA','TIA1'],
    ['PURA',np.nan,'Tubulin','Phalloidin',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'TOMM20'],
    ['DAPI']*12], 
    columns=['A','B','C','D','E','F','G','H','I','J','K','L'],
    index=['ch4', 'ch2', 'ch3','ch1'])

dnls_opera_markers = ['G3BP1','NONO','SQSTM1','PSD95','NEMO','GM130','NCL','LSM14A','TDP43','ANXA11','PEX14','mitotracker',
                      'FMRP','SON','KIF5A','CLTC','DCP1A','Calreticulin','FUS','HNRNPA1','PML','LAMP1','SNCA','TIA1',
                      'PURA','Tubulin','Phalloidin','TOMM20','DAPI']

dnls_opera_cell_lines_for_disp = {f'{cell_line}_{cond}':f'{cell_line}_{cond}' 
                            for cell_line in dnls_opera_cell_lines for cond in dnls_opera_cell_lines_to_cond[cell_line] }

dnls_opera_marker_info = pd.DataFrame([[['ch4']]*12 + [['ch2']]*12 + [['ch3']]*4,
                                       [['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['J'],['K'],['L'], 
                                        ['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['J'],['K'],['L'], 
                                        ['A'],['C'],['D'],['L']]], 
                                      index=['Antibody','panel'],
                                      columns =['G3BP1','NONO','SQSTM1','PSD95','NEMO','GM130','NCL',
                                                'LSM14A','TDP43','ANXA11','PEX14','mitotracker',
                                                'FMRP','SON','KIF5A','CLTC','DCP1A','Calreticulin',
                                                'FUS','HNRNPA1','PML','LAMP1','SNCA','TIA1',
                                                'PURA','Tubulin','Phalloidin','TOMM20']).T #order here is important!!
dnls_opera_line_colors = {
    'dNLS Untreated': colorblind_palette[4],
    'dNLS DOX': colorblind_palette[2]
}

dnls_opera_lines_order = dnls_opera_line_colors.keys()

dnls_opera_custom_palette = [dnls_opera_line_colors[line] for line in dnls_opera_lines_order]

dnls_opera_expected_dapi_raw = 250*len(dnls_opera_panels.columns)

dnls_opera_cell_lines_to_reps = {
    'dNLS': [f'rep{i}' for i in range(1,4)],
}

# Three reps in DOX condition
dnls_opera_reps = [f'rep{i}' for i in range(1,4)]



# ----------------------------------------- #
# Alyssa Coyne - pilot data (only 4 markers)
# ----------------------------------------- #
AC_panels = pd.DataFrame([['DCP1A'],
                          ['Map2'],
                          ['TDP43'],
                          ['DAPI']], columns=['A'],
            index=['Cy5', 'mCherry', 'GFP','DAPI'])

AC_markers = ['DCP1A','Map2','TDP43','DAPI']

AC_reps = ['rep1', 'rep2', 'rep3', 'rep4', 'rep5', 'rep6', 'rep7', 'rep8', 'rep9', 'rep10']

AC_marker_info = pd.DataFrame([[['Cy5']]*1 + [['mCherry']]*1 + [['GFP']]*1,
                          [['A'],['A'],['A']]], index=['Antibody','panel'],
                         columns = ['DCP1A','Map2','TDP43']).T  #order here is important - taken from Lena's sheet

AC_cell_lines = ['Controls','sALS_Negative_cytoTDP43','sALS_Positive_cytoTDP43','c9orf72_ALS_patients']

AC_cell_lines_to_cond = {
                    'Controls':['Untreated'], 
                    'sALSNegativeCytoTDP43':['Untreated'], 
                    'sALSPositiveCytoTDP43':['Untreated'],
                    'c9orf72ALSPatients':['Untreated']}

AC_cell_lines_for_disp = {'Controls_Untreated':'Controls', 
                       'sALSNegativeCytoTDP43_Untreated':'sALS_Negative_cytoTDP43', 
                       'sALSPositiveCytoTDP43_Untreated':'sALS_Positive_cytoTDP43', 
                       'c9orf72ALSPatients_Untreated':'c9orf72_ALS_patients'
                      } 

AC_colorblind_palette = sns.color_palette('colorblind')
AC_line_colors = {
    'c9orf72ALSPatients': AC_colorblind_palette[8],
    'sALSPositiveCytoTDP43': AC_colorblind_palette[5],
    'sALSNegativeCytoTDP43': AC_colorblind_palette[6],
    'Controls': AC_colorblind_palette[9]
}
AC_lines_order = AC_line_colors.keys()
AC_custom_palette = [AC_line_colors[line] for line in AC_lines_order]
AC_expected_dapi_raw = 10

AC_cell_lines_to_reps = {
    'c9orf72ALSPatients': [f'rep{i}' for i in range(1,4)],
    'sALSPositiveCytoTDP43': [f'rep{i}' for i in range(1,10)],
    'sALSNegativeCytoTDP43': [f'rep{i}' for i in range(1,3)],
    'Controls': [f'rep{i}' for i in range(1,7)]
}



# ----------------------------------------- #
## Coyne - NEW DATA (full organelomics)
# ----------------------------------------- #

AC_panels_new = pd.DataFrame([['TDP43', 'LaminB1', 'G3BP1', 'FUS', 'TIA1', 'SCNA', 'SQSTM1', 'NEMO', 'GM130', 'NONO', 'hnRNPA1', 'hnRNPA2B1'],
                            ['Map2', 'Nup62', 'PURA', np.nan, 'Nup98', 'Nup153', 'PSD95', 'Phalloidin', np.nan, np.nan, np.nan, np.nan],
                            ['DCP1A', 'POM121', 'KIF5A', 'FMRP', 'TOMM20', 'ANXA11', 'Lamp1', 'NCL', 'Calreticulin', 'CLTC', 'EEA1', 'Calnexin'],
                            ['DAPI']*12], columns=['A','B','C','D','E','F','G','H','I','J','K','L'],
            index=['Cy5', 'mCherry', 'GFP','DAPI'])

AC_reps_new = ['rep1', 'rep2']

AC_marker_info_new = pd.DataFrame([[['Cy5']]*12 + [['mCherry']]*7 + [['GFP']]*12,
                          [['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['J'],['K'],['L'],
                          ['A'],['B'],['C'],['E'],['F'],['G'],['H'],
                          ['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['J'],['K'],['L']]], index=['Antibody','panel'],
                         columns = ['TDP43', 'LaminB1', 'G3BP1', 'FUS', 'TIA1', 'SCNA', 'SQSTM1', 'NEMO', 'GM130', 'NONO', 'hnRNPA1', 'hnRNPA2B1',
                         'Map2', 'Nup62', 'PURA', 'Nup98', 'Nup153', 'PSD95', 'Phalloidin',
                         'DCP1A', 'POM121', 'KIF5A', 'FMRP', 'TOMM20', 'ANXA11', 'Lamp1', 'NCL', 'Calreticulin', 'CLTC', 'EEA1', 'Calnexin'
                         ]).T  #order here is important - taken from Lena's sheet

AC_markers_new = ['ANXA11','DCP1A','FUS','hnRNPA1','LaminB1','NCL','Nup153','POM121','SCNA','TIA1',
                  'Calnexin','CLTC','EEA1','G3BP1','hnRNPA2B1','Lamp1','NEMO','Nup62','PSD95','SQSTM1','TOMM20',
                  'Calreticulin','DAPI','FMRP','GM130','KIF5A','Map2','NONO','Nup98','PURA','TDP43','Phalloidin']

AC_cell_lines_new = ['C9-CS2YNL','C9-CS8RFT','Ctrl-EDi029','SALSNegative-CS0ANK','SALSNegative-CS6ZU8',
                    'SALSPositive-CS4ZCD','C9-CS7VCZ','Ctrl-EDi022','Ctrl-EDi037','SALSNegative-CS0JPP',
                    'SALSPositive-CS2FN3','SALSPositive-CS7TN6']

AC_cell_lines_to_cond_new = {cell_line:['Untreated'] for cell_line in AC_cell_lines_new}

AC_cell_lines_for_disp_new = {f'{cell_line}_Untreated':cell_line.replace('-','\n') for cell_line in AC_cell_lines_new} 

group_order = ['C9', 'Ctrl', 'SALSNegative', 'SALSPositive']
group_to_lines = defaultdict(list)
for cl in AC_cell_lines_new:
    for group in group_order:
        if cl.startswith(group):
            group_to_lines[group].append(cl)
            break

base_colors = sns.color_palette('colorblind', len(group_order))
group_to_color = dict(zip(group_order, base_colors))

shade_factors = [0.8, 0.6, 0.4]

# Build color dictionary
AC_line_colors_new = {}
for group in group_order:
    base = group_to_color[group]
    lines = group_to_lines[group]
    for i, cl in enumerate(lines):
        factor = shade_factors[i % 3]
        shade = tuple(c * factor + (1 - factor) for c in base)  # blend toward white
        AC_line_colors_new[cl] = shade

AC_colorblind_palette_new = sns.color_palette('colorblind', n_colors=len(AC_cell_lines_new))

AC_lines_order_new = natsort.natsorted(AC_line_colors_new.keys())
AC_custom_palette_new = [AC_line_colors_new[line] for line in AC_lines_order_new]

AC_expected_dapi_raw_new = 5*12


# ----------------------------------------- #
# NIH Ward Lab iNDI Day 8 Neurons
# ----------------------------------------- #

NIH_d8_panels = pd.DataFrame([
             ['DAPI']*11,
             ['FMRP', 'CLTC', 'DCP1A', 'ANAX11', 'PURA', 'PEX14', 'G3BP1', 'LAMP1', 'Phalloidin', 'P54', 'NCL'],
             ['SQSTM1', 'PSD95', 'NEMO', 'SNCA', 'TIA1', 'PML', 'KIF5A', 'Calreticulin', 'GM130', np.nan, 'FUS'],
             ['TUJ1', 'TUJ1', 'TUJ1', 'TUJ1', 'TUJ1', 'TUJ1', 'TUJ1', 'TUJ1', 'TOMM20', 'TDP43', 'MitoTracker']],
             columns=['A','B','C','D','E','F','G','H','I','J','K'],)

NIH_d8_markers = ['ANAX11', 'DAPI', 'SNCA', 'TUJ1', 'DCP1A', 'NEMO', 
                  'CLTC', 'PSD95', 'P54', 'TDP43', 'GM130', 
                  'Phalloidin', 'TOMM20', 'Calreticulin', 'LAMP1', 
                  'FMRP', 'SQSTM1', 'G3BP1', 'KIF5A', 'PEX14', 'PML', 
                  'PURA', 'TIA1', 'FUS', 'MitoTracker', 'NCL']

NIH_d8_marker_info = pd.DataFrame([[['']]*26,
                                   [['D'], ['K'], ['D'], ['H'], ['C'], ['C'], ['B'], ['B'], ['J'], ['J'], ['I'], ['I'],
                                    ['I'], ['H'], ['H'], ['A'], ['A'], ['G'], ['G'], ['F'], ['F'], ['E'], ['E'], ['K'], ['K'], ['K']]],
                                  index=['Antibody','panel'],
                                  columns = ['ANAX11', 'DAPI', 'SNCA', 'TUJ1', 'DCP1A', 'NEMO', 'CLTC', 'PSD95', 'P54',
                                             'TDP43', 'GM130', 'Phalloidin', 'TOMM20', 'Calreticulin', 'LAMP1', 'FMRP',
                                             'SQSTM1', 'G3BP1', 'KIF5A', 'PEX14', 'PML', 'PURA', 'TIA1', 'FUS', 'MitoTracker', 'NCL']).T  

NIH_d8_cell_lines = ['WT']

NIH_d8_cell_lines_to_cond = {'WT':['Untreated','stress']}

NIH_d8_cell_lines_for_disp = {'WT_stress':'WT stress', 
                              'WT_Untreated':'WT Untreated'}

NIH_d8_reps = ['rep1', 'rep2', 'rep3', 'rep4', 'rep5', 'rep6', 'rep7', 'rep8']

NIH_d8_colorblind_palette = sns.color_palette('colorblind')

NIH_d8_line_colors = {
    'WT Untreated': NIH_d8_colorblind_palette[3],
    'WT stress': NIH_d8_colorblind_palette[4]
}

NIH_d8_lines_order = NIH_d8_line_colors.keys()
NIH_d8_custom_palette = [NIH_d8_line_colors[line] for line in NIH_d8_lines_order] 
NIH_d8_expected_dapi_raw = 25*len(NIH_d8_panels.columns)


# ----------------------------------------- #
# Opera 18 days
# ----------------------------------------- #

opera18days_panels = pd.DataFrame([
    ['TOMM20','TDP43','FMRP','PSD95','NEMO','GM130','FUS','SNCA','LAMP1','Tubulin','VDAC1', 'PML',],
    ['PURA',np.nan,'Phalloidin','CLTC','DCP1A','KIF5A','NCL','ANXA11','Calreticulin','PSPC1',np.nan,'PEX14'],
    ['G3BP1','AGO2','SQSTM1',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'HNRNPA1','NONO','mitotracker'],
    ['DAPI']*12],
    columns=['A','B','C','D','E','F','G','H','I','J', 'K', 'L'],
    index=['ch2', 'ch3', 'ch4','ch1'])

opera18days_markers = ['G3BP1','NONO','SQSTM1','PSD95','NEMO',
                       'GM130','NCL','ANXA11','Calreticulin','mitotracker',
                        'KIF5A','TDP43','FMRP','CLTC','DCP1A',
                        'TOMM20','FUS','SNCA','LAMP1','PML',
                        'PURA','Phalloidin', 'PEX14','DAPI',
                        'Tubulin', 'PSPC1', 'VDAC1', 'AGO2', 'HNRNPA1']

opera18days_marker_info = pd.DataFrame([[['ch2']]*12 + [['ch3']]*10 + [['ch4']]*6,
                          [['A'],['B'],['C'],['D'],['E'],['F'],['G'],['H'],['I'],['J'],['K'], ['L'],
                           ['A'],['C'],['D'],['E'], ['F'],['G'],['H'],['I'],['J'],['L'],
                          ['A'],['B'],['C'],['J'],['K'], ['L']]], index=['Antibody','panel'],
                         columns = ['TOMM20','TDP43','FMRP','PSD95','NEMO','GM130','FUS','SNCA',
                                    'LAMP1','Tubulin','VDAC1',
                                    'PML','PURA','Phalloidin','CLTC',
                                    'DCP1A','KIF5A','NCL','ANXA11','Calreticulin','PSPC1','PEX14',
                                    'G3BP1','AGO2','SQSTM1','HNRNPA1','NONO','mitotracker']).T  #order here is important - 

opera18days_cell_lines = ['WT', 'FUSHomozygous', 'FUSHeterozygous','FUSRevertant'] 

opera18days_cell_lines_to_cond = {'WT':['Untreated'],
                      'FUSHomozygous':['Untreated'],
                      'FUSHeterozygous':['Untreated'],
                      'FUSRevertant': ['Untreated']}

opera18days_cell_lines_for_disp = {f'{k}_{c}':f'{k}_{c}' for k,conds in opera18days_cell_lines_to_cond.items() for c in conds}

opera18days_reps = ['rep1','rep2']

opera18days_colorblind_palette = sns.color_palette('colorblind')

opera_colorblind_palette = sns.color_palette('colorblind')

opera18days_line_colors = {f'{k} {c}': opera_colorblind_palette[count % len(opera_colorblind_palette)] 
                           for count, (k, c) in enumerate((k, c) 
                                                          for k, conds in opera18days_cell_lines_to_cond.items() 
                                                          for c in conds)}


opera18days_lines_order = opera18days_line_colors.keys()
opera18days_custom_palette = [opera18days_line_colors[line] for line in opera18days_line_colors]
opera18days_expected_dapi_raw = 1200
