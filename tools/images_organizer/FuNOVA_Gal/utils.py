import logging
import os
import shutil
import shlex

import tools.images_organizer.FuNOVA.interactive_umap.config_funova as config
from tools.images_organizer.FuNOVA.interactive_umap.config_funova import FILE_EXTENSION, LOGGING_PATH, SRC_ROOT_PATH

def __get_batch(plate):
    """
    Get the batch name from the plate number.

    Parameters:
        plate (int): Plate number.

    Returns:
        str: Batch name.
    """
    return f"Batch{plate}"

def __get_condition(stress):
    """
    Determine the condition based on the stress value.

    Parameters:
        stress (int): Stress value (1 for Stress, 0 for Untreated).

    Returns:
        str: Condition ('Stress' or 'Untreated').
    """
    return 'stress' if stress == 1 else 'Untreated'


def __get_rep(column):
    """
    Determine the replication number based on the column value.

    Parameters:
        column (int): Column value.

    Returns:
        int: Replication number (2 for even columns, 1 for odd columns).
    """
    return 2 if column % 2 == 0 else 1

def __get_cell_line_id(row):
    """
    Determine the unique identifier for a cell line.

    Parameters:
        row (pd.Series): A pandas Series representing a row of a DataFrame. 
                         The row should contain at least the keys 'CellLine' 
                         and 'PatientID'.

    Returns:
        str: A string combining the 'CellLine' and 'PatientID' values, 
             separated by an underscore ('_').
    """
    return f"{row['CellLine']}_{row['PatientID']}"

def get_folders_to_handle(root_path):
    """
    Get all folders matching the names specified in config.FOLDERS within the given root path.
    The returned folder paths are relative to the root path.

    Parameters:
        root_path (str): The root path to search for folders.

    Returns:
        list: A list of relative paths to folders matching the specified names in config.FOLDERS.
    """
    try:
        if config.FOLDERS is not None:
            # Get all folders matching the specified names in config.FOLDERS
            matching_folders = []
            for dirpath, dirnames, _ in os.walk(root_path):
                for folder in dirnames:
                    if folder in config.FOLDERS:
                        # Append relative path
                        relative_path = os.path.relpath(os.path.join(dirpath, folder), root_path)
                        matching_folders.append(relative_path)
            return matching_folders
        
        # If FOLDERS is not specified, return all folders in the root path (relative paths)
        logging.info(f"'FOLDERS' wasn't specified in the config file. Returning all folders in {root_path}.")
        return [
            os.path.relpath(os.path.join(root_path, f), root_path)
            for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))
        ]
    
    except Exception as e:
        logging.error(f"An error occurred while retrieving folders: {e}")
        return []


def init_logging(logging_path=LOGGING_PATH):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[
                            logging.FileHandler(logging_path),
                            logging.StreamHandler()
                        ])
    
def create_folder_structure_dict(metadata):
    """
    Create a dictionary representation of the folder structure based on the hierarchical order.

    Parameters:
        metadata (pd.DataFrame): The metadata DataFrame containing image information.

    Returns:
        dict: A nested dictionary representing the folder structure.
    """
    folder_structure = {}

    for _, row in metadata.iterrows():
        # Extract hierarchical levels
        batch = __get_batch(row['Plate'])
        cell_line = __get_cell_line_id(row)
        panel = f"panel{row['Panel']}"
        condition = __get_condition(row['Stress'])
        rep = f"rep{__get_rep(row['Column'])}"
        marker = row['Function']

        # Build the hierarchical dictionary
        if batch not in folder_structure:
            folder_structure[batch] = {}
        if cell_line not in folder_structure[batch]:
            folder_structure[batch][cell_line] = {}
        if panel not in folder_structure[batch][cell_line]:
            folder_structure[batch][cell_line][panel] = {}
        if condition not in folder_structure[batch][cell_line][panel]:
            folder_structure[batch][cell_line][panel][condition] = {}
        if rep not in folder_structure[batch][cell_line][panel][condition]:
            folder_structure[batch][cell_line][panel][condition][rep] = {}
        if marker not in folder_structure[batch][cell_line][panel][condition][rep]:
            folder_structure[batch][cell_line][panel][condition][rep][marker] = []

        # Append the image to the appropriate location
        folder_structure[batch][cell_line][panel][condition][rep][marker].append({'ImageName': row['ImageName'],'Path': row['Path']})

    return folder_structure

def validate_folder_structure(folder_structure):
    """
    Validate the number of files and folders in the hierarchical folder structure.

    Parameters:
        folder_structure (dict): A nested dictionary representing the folder structure.

    Returns:
        dict: A summary containing the number of folders and files.
    """
    def recursive_count(structure):
        folder_count = 0
        file_count = 0

        for key, value in structure.items():
            if isinstance(value, dict):  # It's a folder
                sub_folder_count, sub_file_count = recursive_count(value)
                folder_count += 1 + sub_folder_count  # Count this folder and its subfolders
                file_count += sub_file_count  # Add files from subfolders
            elif isinstance(value, list):  # It's a list of files
                file_count += len(value)  # Count files in this folder

        return folder_count, file_count

    # Get the total folder and file count
    total_folders, total_files = recursive_count(folder_structure)

    # Return the summary
    return {
        "Total Folders": total_folders,
        "Total Files": total_files
    }

def create_folders_from_structure(folder_structure, root_path):
    """
    Create folders based on the hierarchical dictionary structure.

    Parameters:
        folder_structure (dict): Nested dictionary representing the folder structure.
        root_path (str): The root directory where the folders should be created.

    Returns:
        None
    """
    def recursive_create_folders(structure, base_path):
        for key, value in structure.items():
            # Construct the folder path
            folder_path = os.path.join(base_path, str(key))
            # Create the folder if it doesn't exist
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            # Recursively create subfolders
            if isinstance(value, dict):
                recursive_create_folders(value, folder_path)

    # Start the recursive folder creation from the root
    recursive_create_folders(folder_structure, root_path)
    
def init_folders(metadata, root_path):
    """
    Initialize the folder structure based on metadata and create the necessary directories.

    Parameters:
        metadata (pd.DataFrame): The metadata DataFrame containing image information.
        root_path (str): The root directory where the folders should be created.

    Returns:
        dict: A dictionary containing the folder structure and validation summary.
    """
    # Step 1: Create the folder structure dictionary
    folder_structure = create_folder_structure_dict(metadata)

    # Step 2: Validate the folder structure
    validation_summary = validate_folder_structure(folder_structure)
    print("Validation Summary:")
    print(f"Total Folders: {validation_summary['Total Folders']}")
    print(f"Total Files: {validation_summary['Total Files']}")

    # Step 3: Create the folders
    create_folders_from_structure(folder_structure, root_path)
    print(f"Folders successfully created in: {root_path}")

    return folder_structure

def copy_files(dst_folder_path, folder_structure, cut_files=False):
    """
    Copy or move files based on the new folder structure logic.

    Parameters:
        dst_folder_path (str): The destination root folder where files will be copied or moved.
        folder_structure (dict): The nested dictionary representing the folder structure and file paths.
        cut_files (bool): If True, move files instead of copying. Defaults to False.

    Returns:
        int: The total number of files copied or moved.
    """
    n_copied = 0

    def recursive_process(structure, current_dst_path):
        nonlocal n_copied

        for key, value in structure.items():
            # Compute the full path for the current level
            level_path = os.path.join(current_dst_path, key)

            if isinstance(value, dict):  # It's a sub-folder
                # Recurse into the sub-folder
                recursive_process(value, level_path)
            elif isinstance(value, list):  # It's a list of files
                for file_info in value:
                    src_path = file_info['Path']
                    file_name = file_info['ImageName']
                    dst_path = os.path.join(level_path, file_name)  # Use level_path for the correct folder level
                    
                    img_name,ext = os.path.splitext(file_name)
                    if ext != FILE_EXTENSION:
                        continue

                    # Copy or move the file
                    try:
                        if cut_files:
                            shutil.move(src_path, dst_path)
                            action = "moved"
                        else:
                            shutil.copy2(src_path, dst_path)
                            action = "copied"
                        n_copied += 1
                        logging.info(f"File {src_path} {action} to {dst_path}")
                    except Exception as e:
                        logging.error(f"Failed to {action} file {src_path} to {dst_path}: {e}", exc_info=True)

    # Start processing from the root of the folder structure
    recursive_process(folder_structure, dst_folder_path)

    return n_copied

def get_expected_number_of_files_to_copy():
    """
    Calculate the expected number of files to copy by counting files in the specified folders.

    Returns:
        int: Total number of files with the specified extension.
    """
    n = 0
    folders = get_folders_to_handle(SRC_ROOT_PATH)  # Get folders to handle

    for f in folders:
        folder_path = os.path.join(SRC_ROOT_PATH, f)

        # Check if the folder exists before proceeding
        if not os.path.exists(folder_path):
            logging.warning(f"Folder does not exist: {folder_path}")
            continue

        # Safely execute the find command
        try:
            command = f"find {shlex.quote(folder_path)} -type f -name '*{FILE_EXTENSION}' | wc -l"
            f_n = os.popen(command).read().strip()
            n += int(f_n)
        except Exception as e:
            logging.error(f"Error processing folder {folder_path}: {e}")

    return n


