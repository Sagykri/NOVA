# Check if all files exist

# Packages 
from multiprocessing import Pool
from datetime import datetime
from glob import glob 
import logging
import pathlib
import os

# Global paths
BATCH_TO_RUN = 'batch3' 
LINE_TO_RUN = 'WT'

BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
INPUT_DIR = os.path.join(BASE_DIR, 'outputs','cell_profiler')
INPUT_DIR_BATCH = os.path.join(INPUT_DIR, BATCH_TO_RUN)
INPUT_DIR_LINE = os.path.join(INPUT_DIR_BATCH, LINE_TO_RUN)
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs','cell_profiler')
LOG_DIR_PATH = os.path.join(OUTPUT_DIR, 'logs')


def set_logging(log_file_path, level=logging.INFO, format=' INFO: %(message)s'):
    formatter = '%(asctime)s %(levelname)-8s %(message)s'
    handlers = [logging.FileHandler(log_file_path + '.log'), logging.StreamHandler()]
    logging.basicConfig(level=level, format=formatter, handlers=handlers, datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(__doc__)
    
    
    return None


def find_marker_folders(batch_path, depth=5):
    """ 
    For a given batch (defined by "batch_path") it "walks" to its subfolders (until "depth" is reached) 
    and returns for every marker a list of relevant paths (AKA, [input_path, output_path] )
    
    Note: Markers are assumend to be always in a given constant "depth" 
    
    
    """

    # Recursively list files and directories up to a certain depth
    depth -= 1
    with os.scandir(batch_path) as input_data_folder:
        for entry in input_data_folder:
            
            if entry.is_dir(): 
                # replace the prefix of the full path 
                output_folder = entry.path.replace(INPUT_DIR, OUTPUT_DIR)
                # create output folder
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
    
            # if that's not a marker directory, recursion...
            if entry.is_dir() and depth > 0:
                yield from find_marker_folders(entry.path, depth)
            
            # if that's a marker directory
            elif depth==0: 
                #logging.info(f"Checking folder: {entry.path}")
                marker_name = os.path.basename(entry.path)
                #skip nucleus 
                if marker_name=='DAPI':
                    continue
                #skip analyzed markers
                if len(os.listdir(entry.path)) == 7:
                    logging.info(f"Marker already analyzed: {entry.path}")
                    continue
                else:
                    # This is a list of arguments, used as the input of analyze_marker()
                    logging.info(f"Empty marker: {entry.path}")
                    yield [entry.path, output_folder]


def main():

    logging.info(f"\n\nStarting to check if output files exist in batch: {INPUT_DIR_BATCH} {INPUT_DIR_LINE}")

    for subfolder in find_marker_folders(batch_path=INPUT_DIR_LINE, depth=4):
        results = find_marker_folders(subfolder)
        logging.info(results)

    return None        

if __name__ == '__main__':
    
    # Define the log file once in the begining of the script
    set_logging(log_file_path=os.path.join(LOG_DIR_PATH, datetime.now().strftime('log_%d_%m_%Y_%H_%M')))
    
    main()

