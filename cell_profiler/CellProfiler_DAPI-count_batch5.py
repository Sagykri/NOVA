import cellprofiler_core.pipeline
import cellprofiler_core.preferences
import cellprofiler_core.utilities.java
import pathlib
import os
import glob
import pandas as pd
import numpy as np

# Set starting directory which has data in subfolders of cell lines, and has this script
base_dir = "/home/labs/hornsteinlab/Collaboration/MOmaps/data/raw/SpinningDisk/batch5"
os.chdir(base_dir) #always start from here

# Prepare cellprofiler requirements
cellprofiler_core.preferences.set_headless()
cellprofiler_core.utilities.java.start_java()


p = pathlib.Path('.')
lines = [e for e in p.iterdir() if e.is_dir()]   #list of cell line directories
print(lines)

# Run cellprofiler pipeline on each set of 100 images separately
for line in lines:
    os.chdir(os.path.join(base_dir,line))  #go inside cell line directory
    panels = [e for e in pathlib.Path('.').iterdir() if e.is_dir()]
    for panel in panels:
        os.chdir(os.path.join(f"{base_dir}/{line}",panel))   #go inside panel directory
        treatments = [e for e in pathlib.Path('.').iterdir() if e.is_dir()]
        for treatment in treatments:
            os.chdir(os.path.join(f"{base_dir}/{line}/{panel}",treatment))   #go inside treatment directory
            reps = [e for e in pathlib.Path('.').iterdir() if e.is_dir()]
            for rep in reps:
                rep_dir = os.path.join(f"{base_dir}/{line}/{panel}/{treatment}",rep)
                os.chdir(rep_dir)                                             #go inside rep directory
                markers = [e for e in pathlib.Path('.').iterdir() if e.is_dir()]
                for marker in markers:
                    if 'DAPI' in marker.parts:    #Path.parts returns a tuple with the path's components
                        os.chdir(os.path.join(f"{base_dir}/{line}/{panel}/{treatment}/{rep}", marker))   #go inside marker directory

                        if not os.path.exists('CellProfiler_DAPI-count'):
                            os.mkdir('CellProfiler_DAPI-count')

                        current_dir = pathlib.Path().absolute()
                        print(current_dir)

                        pipeline = cellprofiler_core.pipeline.Pipeline()
                        pipeline.load(os.path.join(f"{base_dir}","CellProfiler_DAPI-count.cppipe"))
                        cellprofiler_core.preferences.set_default_output_directory(f"{current_dir}/CellProfiler_DAPI-count")

                        file_list = []

                        for x in os.listdir(current_dir):
                            filename, ext = os.path.splitext(x)
                            if ext == '.tif':
                                file_list.append(pathlib.Path(x).absolute())    #CP requires absolute file paths

                        files = [file.as_uri() for file in file_list]
                        #print(files)
                        pipeline.read_file_list(files)
                        output_measurements = pipeline.run()     #overwrites any output that is already there
                    os.chdir('..')
                os.chdir('..')
            os.chdir('..')
        os.chdir('..')
    os.chdir('..')    

# Extract number of nuclei per well
DAPI_count = pd.DataFrame()

for line in lines:
    os.chdir(os.path.join(base_dir,line))  #go inside cell line directory
    panels = [e for e in pathlib.Path('.').iterdir() if e.is_dir()]
    for panel in panels:
        os.chdir(os.path.join(f"{base_dir}/{line}",panel))   #go inside panel directory
        treatments = [e for e in pathlib.Path('.').iterdir() if e.is_dir()]
        for treatment in treatments:
            os.chdir(os.path.join(f"{base_dir}/{line}/{panel}",treatment))   #go inside treatment directory
            reps = [e for e in pathlib.Path('.').iterdir() if e.is_dir()]
            for rep in reps:
                rep_dir = os.path.join(f"{base_dir}/{line}/{panel}/{treatment}",rep)
                os.chdir(rep_dir)                                #go inside rep directory
                markers = [e for e in pathlib.Path('.').iterdir() if e.is_dir()]
                for marker in markers:
                    if 'DAPI' in marker.parts:    #Path.parts returns a tuple with the path's components
                        output_dir = os.path.join(f"{base_dir}/{line}/{panel}/{treatment}/{rep}/{marker}","CellProfiler_DAPI-count")
                        os.chdir(output_dir)
                        print(os.getcwd())

                        df = pd.read_csv('DAPI-count_nucleus.csv')
                        #print(df.shape)
                        max_value = df.groupby('ImageNumber')['ObjectNumber'].max()    
                        count = pd.DataFrame()
                        count[f'{panel}'] = max_value
                        count['cell_line'] = [f'{line}_{treatment}_{rep}']*len(count[f'{panel}']) 
                        #print(count.head())
                        count = count.set_index(count['cell_line'], append = True, drop = True) 
                                    #assign again to count because it creates copy!
                                    #cannot merge on duplicate index values, so need image-number also as index
                        #count = count.set_index(count[f'{line}_{rep}'], drop = True)
                        count = count.drop('cell_line', axis = 1)
                        #print(count.head())
                        
                        if DAPI_count.empty:
                            DAPI_count = count.copy(deep=True)
                        elif f'{panel}' not in list(DAPI_count):
                            DAPI_count[f'{panel}'] = count[f'{panel}']
                        else:
                            DAPI_count = pd.concat([DAPI_count, count])
                        
                        #print(DAPI_count.head())
                    os.chdir('..')
                os.chdir('..')
            os.chdir('..')
        os.chdir('..')
    os.chdir('..')

DAPI_count.to_csv('CP_DAPI-count.csv')

