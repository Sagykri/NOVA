import pandas as pd
import seaborn as sns
import numpy as np


funova_cell_lines = ['Control-1001733', 'Control-1017118', 'Control-1025045', 'Control-1048087', 
                     'C9orf72-HRE-1008566', 'C9orf72-HRE-981344', 
                     'TDP--43-G348V-1057052', 'TDP--43-N390D-1005373']

funova_cell_lines_to_cond = {
    'Control-1001733': ['Untreated', 'stress'],
    'Control-1017118': ['Untreated', 'stress'],
    'Control-1025045': ['Untreated', 'stress'],
    'Control-1048087': ['Untreated', 'stress'],
    'C9orf72-HRE-1008566': ['Untreated', 'stress'],
    'C9orf72-HRE-981344': ['Untreated', 'stress'],
    'TDP--43-G348V-1057052': ['Untreated', 'stress'],
    'TDP--43-N390D-1005373': ['Untreated', 'stress']
}

funova_markers = ['Autophagy', 'DAPI', 'impaired-Autophagosome', 'UPR-ATF4', 'UPR-ATF6',
                  'UPR-IRE1a', 'Ubiquitin-levels', 'DNA-damage-P53BP1', 'Neuronal-activity',
                  'Necroptosis-HMGB1', 'Necrosis', 'DNA-damage-pH2Ax', 'Parthanatos-early',
                  'Cytoskeleton', 'Stress-initiation', 'mature-Autophagosome',
                  'Nuclear-speckles-SON', 'TDP-43', 'Nuclear-speckles-SC35',
                  'Splicing-factories', 'Aberrant-splicing', 'Parthanatos-late',
                  'Protein-degradation', 'Senescence-signaling', 'Apoptosis', 'Necroptosis-pMLKL']


funova_reps = ['rep1', 'rep2',]


temp_panel_data = {
    "A": ["DAPI", "Stress-initiation", "mature-Autophagosome", "Cytoskeleton"],
    "B": ["DAPI", "UPR-IRE1a", "Ubiquitin-levels"],
    "C": ["DAPI", "UPR-ATF4", "UPR-ATF6"],
    "D": ["DAPI", "Autophagy", "impaired-Autophagosome"],
    "E": ["DAPI", "Parthanatos-late", "Aberrant-splicing"],
    "F": ["DAPI", "Splicing-factories", "Nuclear-speckles-SC35"],
    "G": ["DAPI", "Nuclear-speckles-SON", "TDP-43"],
    "H": ["DAPI", "Parthanatos-early", "DNA-damage-pH2Ax"],
    "I": ["DAPI", "Necrosis", "Necroptosis-HMGB1"],
    "J": ["DAPI", "Neuronal-activity", "DNA-damage-P53BP1"],
    "K": ["DAPI", "Apoptosis", "Necroptosis-pMLKL"],
    "L": ["DAPI", "Protein-degradation", "Senescence-signaling"]
}

# Convert the dictionary to a DataFrame
funova_panels = pd.DataFrame.from_dict(temp_panel_data, orient="index").T

funova_marker_info = pd.DataFrame(
    [
        [[] for _ in range(26)],  # 26 empty lists for the 'Antibody' row
        [
            ['D'], 
            ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'], 
            ['D'], 
            ['C'], ['C'], ['B'], ['B'], ['J'], ['J'], 
            ['I'], ['I'], ['H'], ['H'], ['A'], ['A'], 
            ['A'], ['G'], ['G'], ['F'], ['F'], ['E'], 
            ['E'], ['L'], ['L'], ['K'], ['K'], 
        ]
    ],
    index=['Antibody', 'panel'],  # Two rows: 'Antibody' and 'panel'
    columns=[
        'Autophagy', 'DAPI', 'impaired-Autophagosome', 'UPR-ATF4', 'UPR-ATF6',
        'UPR-IRE1a', 'Ubiquitin-levels', 'Neuronal-activity', 'DNA-damage-P53BP1',
        'Necrosis', 'Necroptosis-HMGB1', 'Parthanatos-early', 'DNA-damage-pH2Ax',
        'Stress-initiation', 'mature-Autophagosome', 'Cytoskeleton', 
        'Nuclear-speckles-SON', 'TDP-43', 'Splicing-factories', 'Nuclear-speckles-SC35',
        'Parthanatos-late', 'Aberrant-splicing', 'Protein-degradation', 'Senescence-signaling',
        'Apoptosis', 'Necroptosis-pMLKL'
    ]
).T


funova_cell_lines_for_disp =  {f'{cell_line}_{cond}':f'{cell_line}_{cond}' 
                            for cell_line in funova_cell_lines for cond in funova_cell_lines_to_cond[cell_line] }

funova_colorblind_palette = sns.color_palette('colorblind')
funova_line_colors = {f'{k} {c}': funova_colorblind_palette[i%len(funova_colorblind_palette)] 
                   for i, (k,conds) in enumerate(funova_cell_lines_to_cond.items())
                   for c in conds}
funova_lines_order = funova_line_colors.keys()
funova_custom_palette = [funova_line_colors[line] for line in funova_lines_order]
funova_expected_dapi_raw = 100