import logging
import random
import os


BASE_DIR = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps')
INPUT_DIR = os.path.join(BASE_DIR,'input','images','processed','spd2','SpinningDisk')


# BATCH_TO_RUN = 'batch8'
# INPUT_DIR_BATCH = os.path.join(INPUT_DIR, BATCH_TO_RUN)


def find_marker_folders(batch_path, depth=5, exclude_DAPI=True):
    """Returns paths of all marker folder in a batch (assumed to be depth of 5 levels)
    works with recursion
    Args:
        batch_path (string):  full path of batch folder
        depth (int, optional): depth of marker sub-folders. Defaults to 5.
        exclude_DAPI (boolean, defualt True): whether to exclude the DAPI folders
        Note: Markers are assumend to be always in a given constant "depth"

    Yields:
        string: a path of marker folder
    """
    
    # Recursively list files and directories up to a certain depth
    depth -= 1
    with os.scandir(batch_path) as input_data_folder:
        
        for entry in sorted(input_data_folder, key=lambda e: e.name):
        
            # if that's not a marker directory, recursion...
            if entry.is_dir() and depth > 0:
                yield from find_marker_folders(entry.path, depth, exclude_DAPI)
            elif not entry.is_dir() and depth > 0:
                continue
            # if that's a marker directory
            elif depth==0: 
                marker_name = os.path.basename(entry.path)
                if (exclude_DAPI) and marker_name=='DAPI':
                    continue
                else:
                    # This is a list of arguments, used as the input of analyze_marker()
                    yield entry.path

def sample_image_names_per_marker(input_data_dir, sample_size=1, raw=False):
    """
    For a given target marker, this function samples file names of images 
    (each image is stored in npy of (n_tiles, 100, 100, 2), AKA target and DAPI marker 
    
    Args:
        input_data_dir (string): full path of marker directory
        Note: "input_data_dir" has to point to a marker directory
        sample_size (int, optional): how  many images to sample. Defaults to 1.
        raw (bool): if to sample raw files or not
    Returns:
        _type_: _description_
    """
    try:
        # This will hold the full path of n images (n is defined by "sample_size") of the marker
        filenames = random.sample(sorted(os.listdir(input_data_dir)), sample_size)
        logging.info(f"\nsample_image_names_per_marker: {input_data_dir}. {sample_size} images per marker.")
    except ValueError:
        npy_size = len(sorted(os.listdir(input_data_dir)))
        filenames = random.sample(sorted(os.listdir(input_data_dir)), npy_size) 
        logging.info(f"\n!!! This marker has less then {sample_size} images. Loaded {input_data_dir}. {npy_size} images per marker.")
        
        
    files_list = []
    # Target marker
    for target_file in filenames:
        filename, ext = os.path.splitext(target_file)
        if not raw:
            if ext == '.npy':
                image_filename = os.path.join(input_data_dir, target_file)
        
                # Add to list
                files_list.append(image_filename)
        
            else:
                logging.info(f"sampled file {target_file} was not a npy. re-sampling.. ")
                continue
        elif raw:
            if ext == '.tif':
                image_filename = os.path.join(input_data_dir, target_file)
        
                # Add to list
                files_list.append(image_filename)
        
            else:
                logging.info(f"sampled file {target_file} was not a tif. re-sampling.. ")
                continue

    return files_list

def sample_images_all_markers(cell_line_path=None, sample_size_per_markers=1, num_markers=26, depth=2, raw=False, rep_count=2,
                              cond_count=2, exclude_DAPI=False, markers_to_include=None):
        """Samples random raw images for a given batch 

        Args:
            cell_line_path (string): path to cell line images
            sample_size_per_markers (int, optional): how many images to sample for each marker. Defaults to 1.
            num_markers (int, optional): how many markers to sample. Defaults to 10.
            raw (bool): if to sample raw files or not

        Returns:
            list: list of paths (strings) 
        """
        sampled_images = []
        sampled_markers = set()
        
        # Get a list of all marker folders
        if raw:
            depth+=2
            num_markers*=rep_count*cond_count
        
        marker_subfolder = find_marker_folders(cell_line_path, depth=depth, exclude_DAPI=exclude_DAPI)
        # Sample n markers, and for each marker, sample k images (where n=num_markers and k=sample_size_per_markers)
        for marker_folder in marker_subfolder:
            if not os.path.isdir(marker_folder):
                continue
            if markers_to_include:
                if os.path.basename(marker_folder) not in markers_to_include:
                    continue
            n_images = 0
            if (len(sampled_markers) < num_markers):
                
                if (n_images<sample_size_per_markers):
                    if 'DAPI' in marker_folder and not raw:
                        sample_size_per_markers *= num_markers
                    sampled_marker_images = sample_image_names_per_marker(marker_folder,
                                                                          sample_size=sample_size_per_markers, 
                                                                          raw=raw)
                    
                    if sampled_marker_images:
                        sampled_images.extend(sampled_marker_images)
                        sampled_markers.add(marker_folder)
                        
                        n_images += 1
                if (n_images==sample_size_per_markers): 
                    continue
            
        logging.info(f"sampled_images: {len(sampled_images)}, sampled_markers: {len(sampled_markers)}")
        return sampled_images

def sample_images_all_markers_all_lines(input_dir_batch=None, _sample_size_per_markers=150, _num_markers=26, 
                                        raw=False, all_conds=False, rep_count=2, cond_count=2, exclude_DAPI=False, markers_to_include=None):
    
    images_paths = []
    
    if input_dir_batch is None:
        raise Exception(f"input argument input_dir_batch is None. ")
    logging.info(f"\n\n[sample_images_all_markers_all_lines]: input_dir_batch:{input_dir_batch}, _sample_size_per_markers:{_sample_size_per_markers}, _num_markers:{_num_markers}")
    
    for cell_line in sorted(os.listdir(input_dir_batch)):
        
        # get the full path of cell line images
        cell_line_path = os.path.join(input_dir_batch, cell_line)
        logging.info(f"\n\ncell_line: {cell_line} {cell_line_path}")
        # Sample markers and then sample images of these markers. The returened value is a list of paths (strings) 
        if not all_conds:
            paths = sample_images_all_markers(cell_line_path, sample_size_per_markers=_sample_size_per_markers, 
                                          num_markers=_num_markers, raw=raw, rep_count=rep_count, cond_count=cond_count, exclude_DAPI=exclude_DAPI, markers_to_include=markers_to_include)
            images_paths.extend(paths)

        else:
            for cond in os.listdir(cell_line_path):
                cond_cell_line_path = os.path.join(cell_line_path, cond)
                paths = sample_images_all_markers(cond_cell_line_path, sample_size_per_markers=_sample_size_per_markers, 
                                        num_markers=_num_markers, depth=1, raw=raw, exclude_DAPI=exclude_DAPI, markers_to_include=markers_to_include)
        
                images_paths.extend(paths)
                
        
    return images_paths



    
