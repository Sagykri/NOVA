import pandas as pd
import seaborn as sns
import numpy as np
import os
import sys
sys.path.insert(1, os.getenv("NOVA_HOME"))
from manuscript.preprocessing_config_AAT_NOVA import PreprocessingBaseConfigAATNOVA_pilot2
config = PreprocessingBaseConfigAATNOVA_pilot2() 

funova_cell_lines = config.CELL_LINES

funova_cell_lines_to_cond = {
    config.CELL_LINES[0]: config.CONDITIONS,
    config.CELL_LINES[1]: config.CONDITIONS,
}

funova_markers = config.MARKERS


funova_reps = config.REPS


temp_panel_data = {
    "A": ["DAPI",  "FK-2", "TDP-43", "SMI32", "Brightfield"],
    "B": ["DAPI","pCaMKIIa", "pDRP1", "Brightfield", "TOMM20"],
    "C": ["DAPI", "ATF4", "pTDP-43", "ATF6", "Brightfield"],
    "D": ["DAPI", "G3BP1", "pAMPK", "pS6", "Brightfield"],
    "E": ["DAPI", "UNC13A", "PAR",  "Calreticulin", "Brightfield"],
    "F": ["DAPI", "p62", "POM121",  "CathepsinD", "Brightfield"],
}

# Convert the dictionary to a DataFrame
funova_panels = pd.DataFrame.from_dict(temp_panel_data, orient="index").T

# build mapping: marker -> list of panels
marker_to_panels = {m: [] for m in funova_markers}
for panel, markers in temp_panel_data.items():
    for m in markers:
        marker_to_panels[m].append(panel)

funova_marker_info = pd.DataFrame(
    [
        [[] for _ in range(len(config.MARKERS))],  # 26 empty lists for the 'Antibody' row
        list(marker_to_panels.values())
    ],
    index=['Antibody', 'panel'],  # Two rows: 'Antibody' and 'panel'
    columns=funova_markers
).T

funova_cell_lines_for_disp =  {f'{cell_line}_{cond}':f'{cell_line}_{cond}' 
                            for cell_line in funova_cell_lines for cond in funova_cell_lines_to_cond[cell_line] }

funova_colorblind_palette = sns.color_palette('colorblind')
funova_line_colors = {f'{k} {c}': funova_colorblind_palette[i%len(funova_colorblind_palette)] 
                   for i, (k,conds) in enumerate(funova_cell_lines_to_cond.items())
                   for c in conds}
funova_lines_order = funova_line_colors.keys()
funova_custom_palette = [funova_line_colors[line] for line in funova_lines_order]

funova_expected_marker_raw = 169
funova_expected_dapi_raw = funova_expected_marker_raw * len(funova_markers)