# Cell profiler analysis 

# Packages 
import cellprofiler_core.pipeline
import cellprofiler_core.preferences
import cellprofiler_core.utilities.java

from multiprocessing import Pool
from datetime import datetime
from glob import glob 
import logging
import pathlib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import sys
from functools import partial

BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
sys.path.insert(1, BASE_DIR)

from src.cell_profiler.CellProfiler_unbiased_analysis import *

# Global paths
BATCH_TO_RUN = 'toxicity2' 

INPUT_DIR = os.path.join(BASE_DIR, 'input','images','raw','SpinningDisk')
INPUT_DIR_BATCH = os.path.join(INPUT_DIR, BATCH_TO_RUN)
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs','cell_profiler')
OUTPUT_DIR_BATCH = os.path.join(OUTPUT_DIR, 'toxicity2')

LOG_DIR_PATH = os.path.join(OUTPUT_DIR, 'logs')

                    
def retrieve_toxicity(input_path):
    
    logging.info(f'Collecting CP measurements for marker: {input_path}')
    
    # collect labels
    drug_dir = pathlib.Path(input_path).parent.resolve()
    drug = os.path.basename(drug_dir)    
    time_dir = pathlib.Path(drug_dir).parent.resolve()
    timepoint = os.path.basename(time_dir)
    
    #combine measurements of all objects
    for file in os.listdir(input_path):
        output_file = file.split('.')[0]
        
        if output_file == 'Image': 
            file_full_name = os.path.join(input_path, file) # Nancy
            matrix = pd.read_csv(file_full_name)
            matrix['drug'] = drug
            matrix['timepoint'] = timepoint
            return matrix

def combine_and_plot(frames):
    data = pd.concat(frames)
    data['live_relative'] = data['Count_live'].div(data['Count_nuclei'])
    data.to_csv(os.path.join(OUTPUT_DIR_BATCH, 'toxicity_CP_output.csv'))
    
    logging.info(f'Saved combined data of CP toxicity analysis in {OUTPUT_DIR_BATCH}')
    
    groups = data.groupby('timepoint')
    for name, group in groups:
        logging.info(f"\n\nGroup name:{name} {group.shape}")
        fig = plt.figure()
        gs = GridSpec(2,1)
        ax1 = fig.add_subplot(gs[0])
        sns.boxplot(data=group, x='drug', y='Count_live', ax = ax1)
        ax1.set_title(f'{name} live cell count')
        ax2 = fig.add_subplot(gs[1])
        sns.boxplot(data = group, x = 'drug', y = 'live_relative', ax = ax2)
        ax2.set_title(f'{name} relative live cell count')
        fig.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR_BATCH, f'boxplot_{name}.png'))
        plt.clf()
        logging.info(f'Saved boxplot of CP toxicity analysis of {name}')

def main():

    logging.info(f"\n\nStarting to run toxicity analysis on batch: {BATCH_TO_RUN}") 
    
    # pipeline_path = os.path.join(BASE_DIR,'src','cell_profiler','pipelines','toxicity2_analysis.cppipe')

    # # create a process pool that uses all cpus
    # with Pool(5) as pool:
    #     results = pool.map(partial(analyze_marker, pipeline_path=pipeline_path), find_marker_folders(batch_path=INPUT_DIR_BATCH, output_dir = OUTPUT_DIR, depth=3, toxicity = True))
    #     for result in results:
    #         logging.info(result)

    #     logging.info("Terminating the java utils and process pool (killing all tasks...)")
    #     # # stop java                
    #     cellprofiler_core.utilities.java.stop_java()
    #     # # forcefully terminate the process pool and kill all tasks
    #     pool.terminate()        

    # logging.info("Starting to collect and plot data")
    from src.cell_profiler.CellProfiler_combine_output import find_marker_folders_output
    
    frames = []
    for sub_folder in find_marker_folders_output(batch_path=OUTPUT_DIR_BATCH, depth=3):
        results = retrieve_toxicity(sub_folder)
        frames.append(results)
    
    combine_and_plot(frames)
    
    return None
       
if __name__ == '__main__':
    
    # Define the log file once in the begining of the script
    set_logging(log_file_path=os.path.join(LOG_DIR_PATH, datetime.now().strftime('log_%d_%m_%Y_%H_%M')))
    
    main()

