# TODO: MOVE TO A DIFFERENT FILE/FOLDER + UTILIZE CONFIGURATION (SAGY WROTE THIS TODO)


import os, shutil
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

from distutils.dir_util import copy_tree
from tools.preprocessing_tools.image_sampling_utils import find_marker_folders

BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
SOURCE_DIR = os.path.join(BASE_DIR,'input','images','raw','SpinningDisk','MOmaps_batch_6_9_images_retaken')

# Get folders under MOmaps_batch_6_9_images_retaken
marker_folders = find_marker_folders(SOURCE_DIR, depth=6, exclude_DAPI=False)

for marker_folder in marker_folders:
    print(f"Source marker folder to copy: {marker_folder}")

    # reformat name of destination folder
    destination_batch_folder = marker_folder.replace('/MOmaps_batch_6_9_images_retaken', '').replace('batch_', 'batch')
    # delte existing content in destination folder
    shutil.rmtree(destination_batch_folder)
    # copy the new content to the destination directory 
    copy_tree(src=marker_folder, dst=destination_batch_folder)
    print(f"\n\ncopied directory content.. \nSOURCE: {marker_folder} \nDESTINATION: {destination_batch_folder}")
    # validate that 100 files exist in destination folder
    assert len(os.listdir(destination_batch_folder) ) == 100, f"number of files: {len(os.listdir(destination_batch_folder))}"


print("\n\nDone!!")