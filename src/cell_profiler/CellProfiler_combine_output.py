## Combine measurements from all lines per marker

#Packages
import pandas as pd 
import os
import pathlib 
from datetime import datetime
import logging

# Global paths
BATCH_TO_RUN = 'microglia_LPS_sort/batch1' 

BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
INPUT_DIR = os.path.join(BASE_DIR, 'outputs','cell_profiler')
INPUT_DIR_BATCH = os.path.join(INPUT_DIR, BATCH_TO_RUN)
OUTPUT_DIR = os.path.join(INPUT_DIR_BATCH, 'combined')

LOG_DIR_PATH = os.path.join(INPUT_DIR, 'logs')

def set_logging(log_file_path, level=logging.INFO, format=' INFO: %(message)s'):
    formatter = '%(asctime)s %(levelname)-8s %(message)s'
    handlers = [logging.FileHandler(log_file_path + '.log'), logging.StreamHandler()]
    logging.basicConfig(level=level, format=formatter, handlers=handlers, datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(__doc__)
    
    return None


# create empty list for each marker 
marker_dict = {'DAPI':[], 'G3BP1':[], 'KIF5A':[], 'PURA':[], 
            'NONO':[], 'TDP43':[], 'CD41':[], 'FMRP':[], 'SQSTM1':[], 'Phalloidin':[], 'PSD95':[], 'CLTC':[],
            'NEMO':[], 'DCP1A':[], 'GM130':[], 'TOMM20':[], 'FUS':[], 'NCL':[], 'SCNA':[], 'ANXA11':[],
            'LAMP1':[], 'Calreticulin':[], 'TIA1':[], 'mitotracker':[], 'PML':[], 'PEX14':[], 'pNFKB':[]}


def retrieve_features(input_path):
    
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
    frames = []
    for file in os.listdir(input_path):
        object_type = file.split('_')[-1]
        object_type = object_type.split('.')[0]
        
        if 'Object' not in str(object_type): 
            continue                           #process only files with object measurements
        
        else:
            file_full_name = os.path.join(input_path, file) # Nancy
            matrix = pd.read_csv(file_full_name)
            logging.info(f'Number of objects of object type {object_type}: {len(matrix.index)}')
            
            #average object measurements per image
            matrix = matrix.groupby(['ImageNumber']).mean()
            matrix['object_type'] = object_type
            logging.info(f'Number of objects after averaging: {len(matrix.index)}')

            frames.append(matrix)
    
    data = pd.concat(frames)
    data['replicate'] = rep
    data['treatment'] = treatment
    data['cell_line'] = cell_line    

    #add the dataframe with all object measurements to the right marker list in the dictionary
    marker_dict[os.path.basename(input_path)].append(data)
    
    return marker_dict
        
              
def find_marker_folders(batch_path, depth=5):
    """ 
    For a given batch (defined by "batch_path") it "walks" to its subfolders (until "depth" is reached) 
    and returns for every marker a list of relevant paths (AKA, [input_path, output_path] )
    
    Note: Markers are assumed to be always in a given constant "depth" 
    
    """

    # Recursively list files and directories up to a certain depth
    depth -= 1
    with os.scandir(batch_path) as input_data_folder:
        for entry in input_data_folder:
            
            if os.path.basename(entry.path) == 'combined':
                continue 
            if os.path.basename(entry.path) == 'plots':
                continue
            
            # if that's not a marker directory, recursion...
            if entry.is_dir() and depth > 0:
                yield from find_marker_folders(entry.path, depth)
            
            # if that's a marker directory
            elif depth==0: 
                marker_name = os.path.basename(entry.path)
                if marker_name=='DAPI':
                    continue
                #skip empty directories
                if not os.listdir(entry.path):
                    continue
                else:
                    # This is a list of arguments, used as the input of extract_cell_profiler_output()
                    yield entry.path


def concatenate_features(marker_dict, output_path):
    """
    Takes a dictionary with markers as key, and separate pandas dataframes containing cell profiler
    measurements per line-treatment-rep combination as values
    
    Per marker, should have 10 average measurements x 2 reps = 20 datapoints per cell line,
    20 x 8 cell lines = 160 rows in output csv (when 10% of data was sampled with CP)
    
    Per marker, should have 8 lines x 2 reps = 16 dataframes in the dictionary
    """
    
    for key, value in marker_dict.items():
        logging.info(f'Concatenating features for marker: {key} with {len(value)} dataframes')
        if len(value)>0: # Nancy, since DAPI is empty.....
            marker_df = pd.concat(value, ignore_index = True)
            #write csv with measurements of all conditions
            marker_df.to_csv(f"{output_path}/{key}_all.csv")


def combine_markers(files_path):
    """
    Create a final csv that contains CP features for all markers, lines, conditions, reps
    and object types
    """
    
    output_frames = []
    
    with os.scandir(files_path) as files_folder:
        for entry in files_folder:
            df = pd.read_csv(entry)
            
            file = os.path.basename(entry.path)
            marker = file.replace("_all.csv","")
            df['marker'] = marker
            
            output_frames.append(df)
    
    df_combined = pd.concat(output_frames)
    df_combined = pd.DataFrame(df_combined)
    df_combined.to_csv(os.path.join(OUTPUT_DIR,'all_markers_all.csv'))
        
def main():
    logging.info(f"\n\nStarting to combine Cell Profiler output of batch: {INPUT_DIR_BATCH}")
    for sub_folder in find_marker_folders(batch_path=INPUT_DIR_BATCH, depth=5):
       results = retrieve_features(sub_folder)

    concatenate_features(results, OUTPUT_DIR)
    combine_markers(OUTPUT_DIR)

if __name__ == '__main__':
    
    # Define the log file once in the begining of the script
    set_logging(log_file_path=os.path.join(LOG_DIR_PATH, datetime.now().strftime('log_%d_%m_%Y_%H_%M')))
    
    main()


