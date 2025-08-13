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


script_name = 'Bar_270725_cyto-nuclear_TDP43.cppipe'
marker_targte = 'TDP43'
OutputFinalLocation = 'New_dNLS_TDP43_300725'

print( "script:", script_name,"MarkerTargte:", marker_targte,"FinalFileLocation: ",OutputFinalLocation)

# Paths to local directories 
OUTPUT_DIR = os.path.join(BASE_DIR, 'cell_profiler', 'outputs', OutputFinalLocation) 
LOG_DIR_PATH = os.path.join(BASE_DIR, 'cell_profiler', 'logs')

# Paths to data to analyse
input_folder_name = 'OPERA_dNLS_6_batches_NOVA_sorted'
DATA_INPUT_DIR = os.path.join(BASE_DIR, 'input', 'images', 'raw', input_folder_name)

pipeline_path = os.path.join(BASE_DIR, 'cell_profiler', 'pipelines','FordNLS_new',script_name)

def main(input_data_batch_dir, use_multiprocessing):

    logging.info(f"\n\nStarting to run Cell Profiler pipeline on batch: {DATA_INPUT_DIR} {BATCH_TO_RUN}") 
    
    if use_multiprocessing:    
        # create a process pool that uses all cpus
        with Pool(10) as pool:
            # call the analyze_marker() function for each marker folder in parallel
            for result in pool.map(partial(cp_utils.analyze_marker_with_filtering, pipeline_path=pipeline_path, dataset_name='OPERA_dNLS_6_batches_NOVA_sorted',input_folder_name=input_folder_name), 
                                cp_utils.find_marker_folders(batch_path=input_data_batch_dir, 
                                                                output_dir=OUTPUT_DIR, 
                                                                depth=5, 
                                                                markers_to_include=[marker_targte])):
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
                                                markers_to_include=[marker_targte]):
            cp_utils.analyze_marker_with_filtering(paths, pipeline_path=pipeline_path, dataset_name='OPERA_dNLS_6_batches_NOVA_sorted',input_folder_name=input_folder_name)
            
        # stop java                
        cellprofiler_core.utilities.java.stop_java()
        
    return None        

if __name__ == '__main__':
    
    # Define the log file once in the begining of the script
    cp_utils.set_logging(log_file_path=os.path.join(LOG_DIR_PATH, datetime.now().strftime('log_%d_%m_%Y_%H_%M')))
    
    use_multiprocessing = True
    
    for BATCH_TO_RUN in ['batch1', 'batch2', 'batch3', 'batch4', 'batch5', 'batch6']:  #['batch1', 'batch2', 'batch3', 'batch4', 'batch5', 'batch6']
        DATA_INPUT_DIR_BATCH = os.path.join(DATA_INPUT_DIR, BATCH_TO_RUN)
        main(DATA_INPUT_DIR_BATCH, use_multiprocessing)
        print("finished batch: ", BATCH_TO_RUN)
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    logging.info(f"\n\nDone!")        

"""
Tasks:
1. conda env 
2. generate new pipeline
3. After the call  
    image_files = collect_image_names_per_marker(input_data_dir)
    we need to filter site images: _get_valid_site_image() - Nancy/Sagy
        load each tif file
        apply rescale_intensity() on it 
        on the rescaled image, filter by brenner: is_image_focused()
        IMPORTANT: check DAPI first, and then target
         
    Then, need to remove the path of these images from "image_files" 
    and continue the extract_cell_profilers_features() 

"""