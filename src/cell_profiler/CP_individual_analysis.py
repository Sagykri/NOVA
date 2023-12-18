## Combine measurements from all lines per marker

#Packages
import pandas as pd 
import os
import pathlib 
from datetime import datetime
import logging
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from functools import partial


BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
sys.path.insert(1, BASE_DIR)

# If importing all from a script, it will also import global variables
from src.cell_profiler.CellProfiler_unbiased_analysis import *
from src.cell_profiler.CellProfiler_combine_output import *

# Global paths
BATCH_TO_RUN = 'batch6' 
MARKER = 'ANXA11'

INPUT_DIR = os.path.join(BASE_DIR, 'input','images', 'raw', 'SpinningDisk')
INPUT_DIR_BATCH = os.path.join(INPUT_DIR, BATCH_TO_RUN)
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs','cell_profiler', 'ANXA11')
OUTPUT_DIR_BATCH = os.path.join(OUTPUT_DIR, BATCH_TO_RUN)
OUTPUT_DIR_PLOTS = os.path.join(OUTPUT_DIR, BATCH_TO_RUN, 'plots')

LOG_DIR_PATH = os.path.join(BASE_DIR, 'outputs','cell_profiler','logs')

# Initialize marker dictionary
marker_dict = {f'{MARKER}':[]}

def get_measurements(input_path, marker = MARKER):
    
    logging.info(f'Collecting object measurements for marker: {MARKER}, {input_path}')
    
    # collect labels
    rep_dir = pathlib.Path(input_path).parent.resolve()
    rep = os.path.basename(rep_dir)    
    treatment_dir = pathlib.Path(rep_dir).parent.resolve()
    treatment = os.path.basename(treatment_dir)
    panel_dir = pathlib.Path(treatment_dir).parent.resolve()
    cell_line_dir = pathlib.Path(panel_dir).parent.resolve()
    cell_line = os.path.basename(cell_line_dir)
    
    #combine measurements of all objects
    for file in os.listdir(input_path):
        
        if marker not in str(file): 
            continue                           #process only files with object measurements
        
        else:
            file_full_name = os.path.join(input_path, file) 
            logging.info(f'Reading {file_full_name}')
            matrix = pd.read_csv(file_full_name)
            logging.info(f'\n length {len(matrix)} \n')
            logging.info(f'matrix keys {matrix.keys()} \n')
            #average object measurements per image
            matrix = matrix.groupby(['ImageNumber']).mean()
            #add labels
            matrix['replicate'] = rep
            matrix['treatment'] = treatment
            matrix['cell_line'] = cell_line   

    #add the dataframe with all object measurements to the right marker list in the dictionary
    marker_dict[os.path.basename(input_path)].append(matrix)
    
    return marker_dict

def plot_CP_features(file_path):
    
    """
    Code for customized plots for individual features; adjust accordingly.
    """
    
    logging.info(f'Starting to plot CP features of {INPUT_DIR_BATCH}')
    matrix = pd.read_csv(file_path)
    matrix = matrix.loc[matrix['treatment'] == 'Untreated'] 
    matrix = matrix.loc[(matrix['cell_line'] == 'WT') | (matrix['cell_line'] == 'FUSHomozygous') | (matrix['cell_line'] == 'FUSHeterozygous') | (matrix['cell_line'] == 'FUSRevertant')]
    
    # Choose and change the features you want to plot
    features_to_plot = ['Intensity_MeanIntensity_ANXA11', 'RadialDistribution_FracAtD_ANXA11_1of4',
                        'Texture_Contrast_ANXA11_3_00_256', 'Texture_Contrast_ANXA11_3_01_256']
    
    for feature in features_to_plot:
        fig = plt.figure()
        sns.boxplot(data=matrix, x='cell_line', y=feature)
        plt.title(f'{MARKER} {feature}')
        fig.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, f'boxplot_{feature}_{BATCH_TO_RUN}.png'))
        plt.clf()
        logging.info(f'Saved boxplot of CP {feature} of {MARKER} in {BATCH_TO_RUN}')

# Specifically for deltaNLS, file structure is different
def get_measurements_deltaNLS(input_path, marker = MARKER):
    
    logging.info(f'Collecting object measurements for marker: {input_path}')
    
    # collect labels
    rep_dir = pathlib.Path(input_path).parent.resolve()
    rep = os.path.basename(rep_dir)    
    treatment_dir = pathlib.Path(rep_dir).parent.resolve()
    treatment = os.path.basename(treatment_dir)
    panel_dir = pathlib.Path(treatment_dir).parent.resolve()
    cell_line_dir = pathlib.Path(panel_dir).parent.resolve()
    cell_line = os.path.basename(cell_line_dir)
    
    #combine measurements of all objects
    for file in os.listdir(input_path):
        
        if marker not in str(file): 
            continue                           
        
        else:
            file_full_name = os.path.join(input_path, file) # Nancy
            print(file, file_full_name)
            matrix = pd.read_csv(file_full_name, sep='\t')
            print(len(matrix))
            print(matrix.keys())
            #average object measurements per image
            matrix = matrix.groupby(['ImageNumber']).mean()
            #add labels
            matrix['replicate'] = rep
            matrix['treatment'] = treatment
            matrix['cell_line'] = cell_line
            
            logging.info(f'Number of objects after averaging: {len(matrix.index)}')
    
    #add the dataframe to the right marker list in the dictionary
    marker_dict[os.path.basename(input_path)].append(matrix)
    
    return marker_dict

def plot_CP_features_deltaNLS(file_path):

    logging.info(f'Starting to plot CP features of {INPUT_DIR_BATCH}')
    matrix = pd.read_csv(file_path)
    matrix = matrix[matrix['cell_line'] == 'TDP43']
    
    # Choose and change the features you want to plot
    features_to_plot = ['ObjectNumber', 'AreaShape_Area', 'Intensity_MeanIntensity_DCP1A', 
                        'Texture_Contrast_DCP1A_3_00_256', 'Texture_Contrast_DCP1A_3_01_256']
    
    for feature in features_to_plot:
        fig = plt.figure()
        sns.boxplot(data=matrix, x='treatment', y=feature)
        plt.title(f'DCP1A {feature}')
        fig.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR_PLOTS, f'boxplot_{feature}.png'))
        plt.clf()
        logging.info(f'Saved boxplot of CP {feature} of DCP1A')


def main():
    # logging.info(f"\n\nStarting individual Cell Profiler of {MARKER} analysis on batch: {INPUT_DIR_BATCH}")

    # pipeline_path = os.path.join(BASE_DIR,'src','cell_profiler','pipelines','231210_ANXA11_CP_analysis.cppipe')

    # with Pool(5) as pool:
    #     # call the analyze_marker() function for each marker folder in parallel
    #     results = pool.map(partial(analyze_marker, pipeline_path=pipeline_path), find_marker_folders(batch_path=INPUT_DIR_BATCH, output_dir = OUTPUT_DIR, individual = True, depth=5))
    #     for result in results:
    #         logging.info(result)
    
    # logging.info("Terminating the java utils and process pool (killing all tasks...)")
    # # # stop java                
    # cellprofiler_core.utilities.java.stop_java()
    # # # forcefully terminate the process pool and kill all tasks
    # pool.terminate()        

    # logging.info("Starting to collect and combine data")
    # for sub_folder in find_marker_folders_output(batch_path=OUTPUT_DIR_BATCH, depth=5):
    #    results = get_measurements(sub_folder)
    # concatenate_features(results, os.path.join(OUTPUT_DIR_BATCH, 'combined'))
    # logging.info('Finished combining output measurements for ')
    
    logging.info("Starting to plot data")
    plot_CP_features(os.path.join(OUTPUT_DIR_BATCH, 'combined', f'{MARKER}_all.csv'))
    logging.info(f'Finished plotting features of {INPUT_DIR_BATCH}')

if __name__ == '__main__':
    
    # Define the log file once in the begining of the script
    set_logging(log_file_path=os.path.join(LOG_DIR_PATH, datetime.now().strftime('log_%d_%m_%Y_%H_%M')))
    
    main()
