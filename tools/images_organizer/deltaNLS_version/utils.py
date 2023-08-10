import logging
import os
import shutil
from datetime import datetime

import config_dnls
from config_dnls import DST_ROOT_PATH, FILE_EXTENSION, KEY_CELL_LINES, KEY_MARKERS, KEY_MARKERS_ALIAS_ORDERED, KEY_REPS, LOGGING_PATH, SRC_ROOT_PATH, KEY_COL_WELLS, KEY_ROW_WELLS
from config_dnls import CONFIG, FOLDERS

def __get_cell_line_and_condition_and_rep_by_index(index):
    assert KEY_CELL_LINES in CONFIG, f"Could not find '{KEY_CELL_LINES}' in config"
    for cell_line in CONFIG[KEY_CELL_LINES].keys():
        conditions = CONFIG[KEY_CELL_LINES][cell_line]
        for condition in conditions:
            rngs = CONFIG[KEY_CELL_LINES][cell_line][condition]
            for rep_index, rng in enumerate(rngs):
                rep = CONFIG[KEY_REPS][rep_index]
                if index >= rng[0] and index <= rng[1]:
                    return cell_line, condition, rep
    
    raise Exception(f"Could not find cell line for index: {index}") 

def __get_cell_line_and_condition_and_rep_by_well(well):
    assert KEY_COL_WELLS in CONFIG, f"Could not find '{KEY_COL_WELLS}' in config"
    assert KEY_ROW_WELLS in CONFIG, f"Could not find '{KEY_ROW_WELLS}' in config"
    well_row = well[0]
    well_col = well[1:]
    cell_line, condition = CONFIG[KEY_COL_WELLS][well_col]
    rep = CONFIG[KEY_ROW_WELLS][well_row]
    rep_number = int(rep.replace('rep',''))-1
    return cell_line, condition, rep_number


def __convert_indx_by_range(indx, min, max):
    if indx % 100 == 0:
        return max
    to_keep_from_indx = indx%100
    to_keep_from_min = min // 100
    if to_keep_from_indx < 10:
        return int(str(to_keep_from_min) + '0' + str(to_keep_from_indx))
    else:
        return int(str(to_keep_from_min) + str(to_keep_from_indx))


def __convert_indx_by_wells(indx, wells):
    index_hundred_range = indx//100 if indx%100 !=0 else indx//100 -1
    print(f"wells: {wells}, indx: {indx}, index_hundred_range: {index_hundred_range}")
    if index_hundred_range >= len(wells): #another edge case, sometimes the last files wells were out of focus
        return None, None
    cur_well = wells[index_hundred_range]
    print(cur_well)
    if cur_well == 'BW': # in this case, the current well was a bad well so we don't want to copy those files
        return None, cur_well
    true_cell_line, true_condition, true_rep = __get_cell_line_and_condition_and_rep_by_well(cur_well)
    min, max = __get_reps_ranges(true_cell_line, true_condition)[true_rep]
    new_indx = __convert_indx_by_range(indx, min, max)
    return new_indx, cur_well

def __get_marker_by_alias(alias, panel):
    assert KEY_MARKERS in CONFIG, f"Could not find '{KEY_MARKERS}' in config"
    assert KEY_MARKERS_ALIAS_ORDERED in CONFIG, f"Could not find '{KEY_MARKERS_ALIAS_ORDERED}' in config"
    
    if alias not in CONFIG[KEY_MARKERS_ALIAS_ORDERED]:
        raise Exception(f"Could not find alias '{alias}' in {KEY_MARKERS_ALIAS_ORDERED}")
    
    marker_indx = CONFIG[KEY_MARKERS_ALIAS_ORDERED].index(alias)
    
    marker = CONFIG[KEY_MARKERS][panel][marker_indx]
    
    return marker
    

def __get_batch(batch):
    return batch


def __get_reps_ranges(cell_line, condition):
    assert KEY_CELL_LINES in CONFIG, f"Could not find '{KEY_CELL_LINES}' in config"
    return CONFIG[KEY_CELL_LINES][cell_line][condition]

def __get_reps_names(cell_line, condition):
    rngs = __get_reps_ranges(cell_line, condition)
    return CONFIG[KEY_REPS][:len(rngs)]

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
    
def __get_dst_path(batch, rep, panel, condition, cell_line, marker, file_name):
    return os.path.join(DST_ROOT_PATH,batch,cell_line, panel, condition, rep, marker, file_name)

def __create_folders_if_needed(batch, panel, cell_lines, conditions, markers):
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
        
            reps = __get_reps_names(cell_line, condition)
            
            for rep in reps:
                rep_folder = os.path.join(condition_folder, rep)
                __create_folder_if_needed(rep_folder)
            
                for marker in markers:
                    if marker is None:
                        continue
                    marker_folder = os.path.join(rep_folder, marker)
                    __create_folder_if_needed(marker_folder)
         
def __get_wells(wells):
    wells = wells.split('-')[1:]
    return wells

def __get_folder_info(folder, replace_wells=False):
    wells = None
    if not replace_wells:
        datestamp, batch, panel = folder.split('_')
    else:
        datestamp, batch, panel, wells = folder.split('_')
        wells = __get_wells(wells)
    return datestamp, batch, panel, wells

def __get_file_info(file):
    _, info = file.split('conf')
    marker_alias, indx = info.split('_')
    return marker_alias, indx


def get_folders_to_handle():
    get_all_folders_in_folder = lambda: [f for f in os.listdir(SRC_ROOT_PATH) if os.path.isdir(os.path.join(SRC_ROOT_PATH, f))]
    
    try:
        if FOLDERS is not None:
            return FOLDERS
        
        return get_all_folders_in_folder()
    except:
        logging.info(f"'FOLDERS' wasn't specified in the config file. Therefore, taking all folders in {SRC_ROOT_PATH}")
        
        return get_all_folders_in_folder()

def init_logging(logging_path=LOGGING_PATH):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Append the timestamp to the file name
    log_file_name = f"log_{timestamp}.txt"
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[
                            logging.FileHandler(os.path.join(LOGGING_PATH,log_file_name)),
                            logging.StreamHandler()
                        ])
    
def init_folders(folder, replace_wells=False):
    folder_path = os.path.join(SRC_ROOT_PATH, folder)
    logging.info(f"Current folder: {folder_path}")
        
    datestamp, batch, panel, wells = __get_folder_info(folder, replace_wells)
    batch = __get_batch(batch)
    panel = __get_panel(panel)
    cell_lines = __get_cell_lines()
    conditions = __get_conditions()
    markers = __get_markers(panel)
        
    __create_folders_if_needed(batch, panel, cell_lines, conditions, markers)
    
    return folder_path, batch, panel, wells

def copy_files(folder_path, panel, batch, cut_files=False, replace_wells=False, wells=None):
    n_copied = 0
    
    files_names = os.listdir(folder_path)
    for f in files_names:
        logging.info(f"[{os.path.join(folder_path, f)}]")
        
        file_name,ext = os.path.splitext(f)
            
        if ext != FILE_EXTENSION:
            continue
            
        marker_alias, indx = __get_file_info(file_name)
        indx = int(indx.replace('s', ''))
            
        try:
            if replace_wells:
                orig_indx = indx
                indx, cur_well = __convert_indx_by_wells(indx, wells)
                if not indx:
                    logging.info(f"[{os.path.join(folder_path, f)}] index: {orig_indx} was skipped since well is {cur_well}")
                    continue
                logging.info(f"[{os.path.join(folder_path, f)}] index: {orig_indx} with well {cur_well} was converted to {indx}")
            matched_cell_line, matched_condition, matched_rep = __get_cell_line_and_condition_and_rep_by_index(indx)
            matched_marker = __get_marker_by_alias(marker_alias, panel)
                
            logging.info(f"[{os.path.join(folder_path, f)}] batch={batch},rep={matched_rep}, panel={panel}, condition={matched_condition},cell_line={matched_cell_line}, marker={matched_marker}")

            file_name = f
            if replace_wells:
                orig_indx_str = f"_s{orig_indx}.tif"
                new_indx_str = f"_s{indx}.tif"
                file_name = file_name.replace(orig_indx_str, new_indx_str)
            dst_path = __get_dst_path(batch, matched_rep, panel, matched_condition, matched_cell_line, matched_marker, file_name)
            src_path = os.path.join(folder_path, f)
                
            if cut_files:
                dst_path_full = shutil.move(src_path, dst_path)
            else:
                dst_path_full = shutil.copy2(src_path, dst_path)
            n_copied += 1
            logging.info(f"[{os.path.join(folder_path, f)}] {src_path} {'moved' if cut_files else 'copied'} to {dst_path_full}")
                
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