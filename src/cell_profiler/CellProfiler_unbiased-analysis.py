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

# Global paths
BATCH_TO_RUN = 'batch6' 
LINE_TO_RUN = 'FUSRevertant'

BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
INPUT_DIR = os.path.join(BASE_DIR, 'input','images','raw','SpinningDisk')
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


def init_cell_profiler():
    
    logging.info("\n\nInitializing Cell Profiler..")

    cellprofiler_core.preferences.set_headless()
    cellprofiler_core.utilities.java.start_java()
    
    pipeline = loading_cellprofiler_pipeline()
    
    return pipeline


def loading_cellprofiler_pipeline():

    logging.info("loading_cellprofiler_pipeline..")

    pipeline_path = os.path.join(BASE_DIR,'src','cell_profiler','pipelines','CellProfiler_unbiased-analysis_minimal.cppipe')
    my_pipeline = cellprofiler_core.pipeline.Pipeline()
    my_pipeline.load(pipeline_path)
    
    logging.info("Finished loading_cellprofiler_pipeline..")
    return my_pipeline


def set_marker_output_folder(path):
    
    logging.info(f"set_marker_output_folder: {path}")
    cellprofiler_core.preferences.set_default_output_directory(path)
    return None


def collect_image_names_per_marker(input_data_dir):
    """
    Note: "input_data_dir" has to point to a marker directory
    For a given marker, this function returns all file names of images of target marker and DAPI marker 
    """
    
    logging.info(f"collect_image_names_per_marker: {input_data_dir}")
    
    # This will hold the names of all ~100 images of the marker
    file_list = []
    # Define rep directory
    rep_dir = pathlib.Path(input_data_dir).parent.resolve()
    # Initialize counter for sampling 
    file_count = 0
    # Target marker
    for file in os.listdir(input_data_dir):
        filename, ext = os.path.splitext(file)
        if ext == '.tif':
            image_filename = os.path.join(input_data_dir, file)
            file_list.append(pathlib.Path(image_filename))    # CP requires absolute file paths
    
            #find the right DAPI file to append
            site = filename.split('_')[-1]
            nucleus_folder = os.path.join(rep_dir, "DAPI")
            nucleus_filepath = glob(f"{nucleus_folder}/*_{site}{ext}")[0]
            file_list.append(pathlib.Path(nucleus_filepath))
            
            #stop at 50% of the data
            file_count += 1
            if file_count == 100:
                files = [file.as_uri() for file in file_list]
                return files
            else:
                continue


def extract_cell_profilers_features(image_files, pipeline):

    logging.info(f"extract_cell_profilers_features")
    logging.info(image_files)
    pipeline.read_file_list(image_files)
    logging.info("\n\nX done pipeline.read_file_list()")
    output_measurements = pipeline.run()     #overwrites any output that is already there
    logging.info("\n\nX done pipeline.run()")
    return None


def analyze_marker(input_and_output_path_list):
        
    global pipeline
    pipeline = init_cell_profiler()
    logging.info(pipeline)
    
    input_data_dir, output_folder = input_and_output_path_list[0], input_and_output_path_list[1]
    logging.info(f"Analyzing marker: {input_data_dir}")

    set_marker_output_folder(path=output_folder)

    image_files = collect_image_names_per_marker(input_data_dir)

    extract_cell_profilers_features(image_files, pipeline)

    return f"\n\nFinished extracting features for {input_data_dir}"


def find_marker_folders(batch_path, depth=4):
    """ 
    For a given batch (defined by "batch_path") it "walks" to its subfolders (until "depth" is reached) 
    and returns for every marker a list of relevant paths (AKA, [input_path, output_path] )
    
    Note: Markers are assumed to be always in a given constant "depth" 
    
    
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
                marker_name = os.path.basename(entry.path)
                #skip nucleus 
                if marker_name=='DAPI':
                    continue
                #skip analyzed markers
                if len(os.listdir(output_folder)) == 7:
                    logging.info(f"Marker already analyzed: {output_folder}")
                    continue
                else:
                    # This is a list of arguments, used as the input of analyze_marker()
                    yield [entry.path, output_folder]
                    


def main():

    logging.info(f"\n\nStarting to run Cell Profiler pipeline on batch: {BATCH_TO_RUN}") 
    logging.info(f"\n\nStarting to run Cell Profiler pipeline on line: {LINE_TO_RUN}")

    # TODO: nancy, optimize, can we call this once and broadcast to all multi-processing pools?
    #global pipeline
    #pipeline = init_cell_profiler()
    
    # create a process pool that uses all cpus
    with Pool(5) as pool:
        # call the analyze_marker() function for each marker folder in parallel
        for result in pool.imap_unordered(analyze_marker, find_marker_folders(batch_path=INPUT_DIR_LINE, depth=4)):
            logging.info(result)

    logging.info("Terminating the java utils and process pool (killing all tasks...)")
    # stop java                
    cellprofiler_core.utilities.java.stop_java()
    # forcefully terminate the process pool and kill all tasks
    pool.terminate()
    return None        

if __name__ == '__main__':
    
    # Define the log file once in the begining of the script
    set_logging(log_file_path=os.path.join(LOG_DIR_PATH, datetime.now().strftime('log_%d_%m_%Y_%H_%M')))
    
    main()

