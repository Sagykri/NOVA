
import logging
import os
import shutil

import config
from config import DST_ROOT_PATH, FILE_EXTENSION, KEY_BATCHES, KEY_CELL_LINES, KEY_MARKERS, KEY_MARKERS_ALIAS_ORDERED, KEY_REPS, LOGGING_PATH, SRC_ROOT_PATH
from config import CONFIG

def __get_subkey_contains_val(key, val):
    assert key in CONFIG, f"Could not find '{key}' in config"
    
    d = CONFIG[key]
    for k in d.keys():
        vals = d[k]
        if val in vals:
            return k
        
    raise Exception(f"Could not find {val} in {key}")

def __get_cell_line_and_condition_by_index(index):
    assert KEY_CELL_LINES in CONFIG, f"Could not find '{KEY_CELL_LINES}' in config"
    for cell_line in CONFIG[KEY_CELL_LINES].keys():
        conditions = CONFIG[KEY_CELL_LINES][cell_line]
        for condition in conditions:
            rng = CONFIG[KEY_CELL_LINES][cell_line][condition]
            if index >= rng[0] and index <= rng[1]:
                return cell_line, condition
    
    raise Exception(f"Could not find cell line for index: {index}") 

def __get_marker_by_alias(alias, panel):
    assert KEY_MARKERS in CONFIG, f"Could not find '{KEY_MARKERS}' in config"
    assert KEY_MARKERS_ALIAS_ORDERED in CONFIG, f"Could not find '{KEY_MARKERS_ALIAS_ORDERED}' in config"
    
    if alias not in CONFIG[KEY_MARKERS_ALIAS_ORDERED]:
        raise Exception(f"Could not find alias '{alias}' in {KEY_MARKERS_ALIAS_ORDERED}")
    
    marker_indx = CONFIG[KEY_MARKERS_ALIAS_ORDERED].index(alias)
    
    marker = CONFIG[KEY_MARKERS][panel][marker_indx]
    
    return marker
    

def __get_batch(plate):
    return __get_subkey_contains_val(KEY_BATCHES, plate)
    
def __get_rep(row):
    return __get_subkey_contains_val(KEY_REPS, row)

def __get_panel(panel):
    return panel

def __get_cell_lines():
    assert KEY_CELL_LINES in CONFIG, f"Could not find '{KEY_CELL_LINES}' in config"
    
    return list(CONFIG[KEY_CELL_LINES].keys())

def __get_conditions():
    assert KEY_CELL_LINES in CONFIG, f"Could not find '{KEY_CELL_LINES}' in config"
    
    cell_lines = CONFIG[KEY_CELL_LINES]
    return {key: list(cell_lines[key].keys()) for key in cell_lines.keys()}

def __get_markers(panel):
    assert KEY_MARKERS in CONFIG, f"Could not find '{KEY_MARKERS}' in config"
    assert panel in CONFIG[KEY_MARKERS], f"Could not find '{panel}' in {CONFIG[KEY_MARKERS]}"
    
    markers = CONFIG[KEY_MARKERS]
    
    return markers[panel]

def __create_folder_if_needed(path):
    if os.path.exists(path) and os.path.isdir(path):
        # Not needed
        return
    
    logging.info(f"Creating folder: {path}")
    os.makedirs(path)
    
def __get_dst_path(batch, rep, panel, condition, cell_line, marker):
    return os.path.join(DST_ROOT_PATH,batch,cell_line, panel, condition, rep, marker)

def __create_folders_if_needed(batch, rep, panel, cell_lines, conditions, markers):
    batch_folder = os.path.join(DST_ROOT_PATH, batch)
    __create_folder_if_needed(batch_folder)
    
    for cell_line in cell_lines:
        cell_line_folder = os.path.join(batch_folder, cell_line)
        __create_folder_if_needed(cell_line_folder)
        
        panel_folder = os.path.join(cell_line_folder, panel)
        __create_folder_if_needed(panel_folder)
        
        for condition in conditions[cell_line]:
            condition_folder = os.path.join(panel_folder, condition)
            __create_folder_if_needed(condition_folder)
        
            rep_folder = os.path.join(condition_folder, rep)
            __create_folder_if_needed(rep_folder)
            
            for marker in markers:
                if marker is None:
                    continue
                marker_folder = os.path.join(rep_folder, marker)
                __create_folder_if_needed(marker_folder)
            
def __get_folder_info(folder):
    datestamp, plate, row, panel_keyword, panel_id = folder.split('_')
    panel = f"{panel_keyword}{panel_id}"
    return datestamp, plate, row, panel

def __get_file_info(file):
    _, info = file.split('conf')
    marker_alias, indx = info.split('_')
    return marker_alias, indx


def get_folders_to_handle():
    get_all_folders_in_folder = lambda: [f for f in os.listdir(SRC_ROOT_PATH) if os.path.isdir(os.path.join(SRC_ROOT_PATH, f))]
    
    try:
        if config.FOLDERS is not None:
            return config.FOLDERS
        
        return get_all_folders_in_folder()
    except:
        logging.info(f"'FOLDERS' wasn't specified in the config file. Therefore, taking all folders in {SRC_ROOT_PATH}")
        
        return get_all_folders_in_folder()

def init_logging(logging_path=LOGGING_PATH):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[
                            logging.FileHandler(logging_path),
                            logging.StreamHandler()
                        ])
    
def init_folders(folder):
    folder_path = os.path.join(SRC_ROOT_PATH, folder)
    logging.info(f"Current folder: {folder_path}")
        
    datestamp, plate, row, panel = __get_folder_info(folder)
    batch = __get_batch(plate)
    rep = __get_rep(row)
    panel = __get_panel(panel)
    cell_lines = __get_cell_lines()
    conditions = __get_conditions()
    markers = __get_markers(panel)
        
    __create_folders_if_needed(batch, rep, panel, cell_lines, conditions, markers)
    
    return folder_path, batch, rep, panel

def copy_files(folder_path, panel, batch, rep):
    n_copied = 0
    
    files_names = os.listdir(folder_path)
    for f in files_names:
        file_name,ext = os.path.splitext(f)
            
        if ext != FILE_EXTENSION:
            continue
            
        marker_alias, indx = __get_file_info(file_name)
        indx = int(indx.replace('s', ''))
            
        try:
            matched_cell_line, matched_condition = __get_cell_line_and_condition_by_index(indx)
            matched_marker = __get_marker_by_alias(marker_alias, panel)
                
            dst_path = __get_dst_path(batch, rep, panel, matched_condition, matched_cell_line, matched_marker)
            src_path = os.path.join(folder_path, f)
                
            dst_path_full = shutil.copy2(src_path, dst_path)
            n_copied += 1
            logging.info(f"[{folder_path}] {f} copied to {dst_path_full}")
                
        except Exception as e:
            logging.error(e, exc_info=True)
            raise
        
    return n_copied

def get_expected_number_of_files_to_copy():
    n = 0
    
    for f in get_folders_to_handle():
        f = os.path.join(SRC_ROOT_PATH, f)
        f_n = os.popen(f"find {f} -type f -name '*{FILE_EXTENSION}' | wc -l").read()
        n += int(f_n)
        
    return n