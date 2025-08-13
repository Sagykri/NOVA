# Cell profiler analysis 

# Packages 
from multiprocessing import Pool
from datetime import datetime
from functools import partial
import logging
import sys
import os

import cellprofiler_core.utilities.java
import cell_profiler_utils as cp_utils


# To be able to import from other scripts under "cell_profiler"
BASE_DIR = os.path.join('/home','projects','hornsteinlab','Collaboration','NOVA')
sys.path.insert(1, BASE_DIR)

input_folder_name = 'AlyssaCoyne'
marker_targte = ['DCP1A'] 
output_folder_name = 'coyne_old_Pb_extracte_F_DCP1A_120825'
pipeline_name = 'PB_feature_extraction_coyne_old_DCPA1.cppipe'

# Paths to data to analyse
DATA_INPUT_DIR = os.path.join(BASE_DIR, 'input', 'images', 'raw',input_folder_name, 'MOmaps_iPSC_patients_TDP43_PB_CoyneLab')
BATCH_TO_RUN = 'batch1' 
#LINE_TO_RUN = 'WT'


# Paths to local directories 
OUTPUT_DIR = os.path.join(BASE_DIR, 'cell_profiler', 'outputs', "cell_profiler_RUNS" , output_folder_name)
LOG_DIR_PATH = os.path.join(BASE_DIR, 'cell_profiler', 'logs')


DATA_INPUT_DIR_BATCH = os.path.join(DATA_INPUT_DIR, BATCH_TO_RUN)

pipeline_path = os.path.join(BASE_DIR, 'cell_profiler', 'pipelines','ForCoyneData', pipeline_name)



def main(use_multiprocessing):

    logging.info(f"\n\nStarting to run Cell Profiler pipeline on batch: {DATA_INPUT_DIR} {BATCH_TO_RUN}") 

    if use_multiprocessing:    
        # create a process pool that uses all cpus
        with Pool(5) as pool:
            # call the analyze_marker() function for each marker folder in parallel
            for result in pool.map(partial(cp_utils.analyze_marker, pipeline_path=pipeline_path), 
                                   cp_utils.find_marker_folders(batch_path=DATA_INPUT_DIR_BATCH, 
                                                                output_dir=OUTPUT_DIR, 
                                                                depth=5, 
                                                                markers_to_include= marker_targte)):
                logging.info(result)
        logging.info("Terminating the java utils and process pool (killing all tasks...)")
        # stop java                
        cellprofiler_core.utilities.java.stop_java()
        # forcefully terminate the process pool and kill all tasks
        pool.terminate()

    else:
        global pipeline
        pipeline = cp_utils.init_cell_profiler(pipeline_path=pipeline_path)
        
        for paths in cp_utils.find_marker_folders(batch_path=DATA_INPUT_DIR_BATCH, 
                                                  output_dir=OUTPUT_DIR, 
                                                  depth=5, 
                                                  markers_to_include=marker_targte):
            
            cp_utils.analyze_marker(paths, pipeline_path=pipeline_path)
            
        # stop java                
        cellprofiler_core.utilities.java.stop_java()
        
    return None        

if __name__ == '__main__':
    
    # Define the log file once in the begining of the script
    cp_utils.set_logging(log_file_path=os.path.join(LOG_DIR_PATH, datetime.now().strftime('log_%d_%m_%Y_%H_%M')))
    
    use_multiprocessing = True
    
    main(use_multiprocessing)
    
    logging.info(f"\n\nDone!")        

