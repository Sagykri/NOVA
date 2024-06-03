import logging
import os
import shutil
from datetime import datetime

from config import Config

class Utils():
    def __init__(self, config:Config):
        self.config = config
        
    def __get_cell_line_and_condition_and_rep_by_index(self, row_col, raise_on_missing_index=True):
        assert self.config.KEY_CELL_LINES in self.config.CONFIG, f"Could not find '{self.config.KEY_CELL_LINES}' in config"
        for cell_line in self.config.CONFIG[self.config.KEY_CELL_LINES].keys():
            conditions = self.config.CONFIG[self.config.KEY_CELL_LINES][cell_line]
            for condition in conditions:
                rngs = self.config.CONFIG[self.config.KEY_CELL_LINES][cell_line][condition]
                for rep_index, rng in enumerate(rngs):
                    rep = self.config.CONFIG[self.config.KEY_REPS][rep_index]
                    # Changed 190324
                    row, col = row_col
                    if int(row) == rng[0] and int(col) == rng[1]:
                        return cell_line, condition, rep
        
        if raise_on_missing_index:
            raise Exception(f"Could not find cell line for row,col: {row_col}") 
        
        logging.warn(f"Could not find cell line for row,col: {row_col}. Skipping since 'raise_on_missing_index' is set to False")
        return (None, )*3

    def __get_marker_by_alias(self, alias, panel):
        assert self.config.KEY_MARKERS in self.config.CONFIG, f"Could not find '{self.config.KEY_MARKERS}' in config"
        assert self.config.KEY_MARKERS_ALIAS_ORDERED in self.config.CONFIG, f"Could not find '{self.config.KEY_MARKERS_ALIAS_ORDERED}' in config"
        
        if alias not in self.config.CONFIG[self.config.KEY_MARKERS_ALIAS_ORDERED]:
            raise Exception(f"Could not find alias '{alias}' in {self.config.KEY_MARKERS_ALIAS_ORDERED}")
        
        marker_indx = self.config.CONFIG[self.config.KEY_MARKERS_ALIAS_ORDERED].index(alias)
        
        marker = self.config.CONFIG[self.config.KEY_MARKERS][panel][marker_indx]
        
        return marker
        

    def __get_batch(self, batch):
        return batch


    def __get_reps_ranges(self, cell_line, condition):
        assert self.config.KEY_CELL_LINES in self.config.CONFIG, f"Could not find '{self.config.KEY_CELL_LINES}' in config"
        return self.config.CONFIG[self.config.KEY_CELL_LINES][cell_line][condition]

    def __get_reps_names(self, cell_line, condition):
        rngs = self.__get_reps_ranges(cell_line, condition)
        return self.config.CONFIG[self.config.KEY_REPS][:len(rngs)]

    def __get_panel(self, panel):
        return panel

    def __get_cell_lines(self):
        assert self.config.KEY_CELL_LINES in self.config.CONFIG, f"Could not find '{self.config.KEY_CELL_LINES}' in config"
        
        return list(self.config.CONFIG[self.config.KEY_CELL_LINES].keys())

    def __get_conditions(self):
        assert self.config.KEY_CELL_LINES in self.config.CONFIG, f"Could not find '{self.config.KEY_CELL_LINES}' in config"
        
        cell_lines = self.config.CONFIG[self.config.KEY_CELL_LINES]
        return {key: list(cell_lines[key].keys()) for key in cell_lines.keys()}

    def __get_markers(self, panel):
        assert self.config.KEY_MARKERS in self.config.CONFIG, f"Could not find '{self.config.KEY_MARKERS}' in config"
        assert panel in self.config.CONFIG[self.config.KEY_MARKERS], f"Could not find '{panel}' in {self.config.CONFIG[self.config.KEY_MARKERS]}"
        
        markers = self.config.CONFIG[self.config.KEY_MARKERS]
        
        return markers[panel]

    def __create_folder_if_needed(self, path):
        if os.path.exists(path) and os.path.isdir(path):
            # Not needed
            return
        
        logging.info(f"Creating folder: {path}")
        os.makedirs(path)
        
    def __get_dst_path(self, batch, rep, panel, condition, cell_line, marker, file_name):
        return os.path.join(self.config.DST_ROOT_PATH,batch,cell_line, panel, condition, rep, marker, f"{self.config.FILENAME_POSTFIX}{file_name}")

    def __create_folders_if_needed(self, batch, panel, cell_lines, conditions, markers):
        batch_folder = os.path.join(self.config.DST_ROOT_PATH, batch)
        self.__create_folder_if_needed(batch_folder)
        
        for cell_line in cell_lines:
            cell_line_folder = os.path.join(batch_folder, cell_line)
            self.__create_folder_if_needed(cell_line_folder)
            
            panel_folder = os.path.join(cell_line_folder, panel)
            self.__create_folder_if_needed(panel_folder)
            
            for condition in conditions[cell_line]:
                condition_folder = os.path.join(panel_folder, condition)
                self.__create_folder_if_needed(condition_folder)
            
                reps = self.__get_reps_names(cell_line, condition)
                
                for rep in reps:
                    rep_folder = os.path.join(condition_folder, rep)
                    self.__create_folder_if_needed(rep_folder)
                
                    for marker in markers:
                        if marker is None:
                            continue
                        marker_folder = os.path.join(rep_folder, marker)
                        self.__create_folder_if_needed(marker_folder)

    def __get_folder_info(self, folder):
        # parent_folder = os.path.dirname(folder)
        # datestamp, pert1, pert2, pert3, n_days, panels = parent_folder.split('_')
        batch = os.path.dirname(folder)#"batch1"
        panel = os.path.basename(folder)
        # lower case the first letter (Panel to panel)
        panel = panel[0].lower() + panel[1:]
        
        return batch, panel

    def __get_file_info(self, file):
        # Changed! 190324
        coord, info = file.split('-')
        r, c = coord[1:3], coord[4:6]
        ch = info[:4]
        return ch, (r,c)


    def get_folders_to_handle(self):
        def filter_subfodlers(path):
            if path in self.config.EXCLUDE_SUB_FOLDERS:
                return False
            if (len(self.config.INCLUDE_SUB_FOLDERS) > 0) and (path not in self.config.INCLUDE_SUB_FOLDERS):
                return False
            return True
            
        def get_subfolders(root):
            return [os.path.join(root, sub) for sub in os.listdir(os.path.join(self.config.SRC_ROOT_PATH, root)) if filter_subfodlers(os.path.join(root,sub))]
        
        def get_processed_subfolders(folders):
            subfolders = []
            for f in folders:
                f_depth = get_subfolders(f)
                subfolders.extend(f_depth)
            return subfolders
        
        try:
            if self.config.FOLDERS is not None:
                folders = self.config.FOLDERS
            else:
                folders = get_subfolders(self.config.SRC_ROOT_PATH)
            return get_processed_subfolders(folders)
        except:
            logging.info(f"'FOLDERS' wasn't specified in the config file. Therefore, taking all folders in {self.config.SRC_ROOT_PATH}")
            
            folders = get_subfolders(self.config.SRC_ROOT_PATH)
            
            return get_processed_subfolders(folders)

    def init_logging(self, logging_path=None):
        if logging_path is None:
            logging_path = self.config.LOGGING_PATH
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Append the timestamp to the file name
        log_file_name = f"log_{timestamp}.txt"
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S",
                            handlers=[
                                logging.FileHandler(os.path.join(logging_path,log_file_name)),
                                logging.StreamHandler()
                            ])
        
    def init_folders(self, folder):
        folder_path = os.path.join(self.config.SRC_ROOT_PATH, folder)
        logging.info(f"Current folder: {folder_path}")
            
        batch, panel = self.__get_folder_info(folder)
        batch = self.__get_batch(batch)
        panel = self.__get_panel(panel)
        cell_lines = self.__get_cell_lines()
        conditions = self.__get_conditions()
        markers = self.__get_markers(panel)
            
        self.__create_folders_if_needed(batch, panel, cell_lines, conditions, markers)
        
        return folder_path, batch, panel

    def copy_files(self, folder_path, panel, batch, cut_files=False, raise_on_missing_index=True):
        n_copied = 0
        
        files_names = os.listdir(folder_path)
        
        for f in files_names:
            logging.info(f"[{os.path.join(folder_path, f)}]")
            
            file_name,ext = os.path.splitext(f)
                
            if ext != self.config.FILE_EXTENSION:
                continue
                
            ch, row_col = self.__get_file_info(file_name)
            
            try:
                matched_cell_line, matched_condition, matched_rep = self.__get_cell_line_and_condition_and_rep_by_index(row_col, raise_on_missing_index=raise_on_missing_index)
                if matched_cell_line is None:
                    # If on_missing_index isn't set to 'raise' and index wasn't find in config, skip
                    continue
                
                matched_marker = self.__get_marker_by_alias(ch, panel)
                    
                logging.info(f"[{os.path.join(folder_path, f)}] batch={batch},rep={matched_rep}, panel={panel}, condition={matched_condition},cell_line={matched_cell_line}, marker={matched_marker}")
                print(f"[{os.path.join(folder_path, f)}] batch={batch},rep={matched_rep}, panel={panel}, condition={matched_condition},cell_line={matched_cell_line}, marker={matched_marker}")

                if matched_marker is None:
                    logging.warn(f"[{os.path.join(folder_path, f)}] batch={batch},rep={matched_rep}, panel={panel}, condition={matched_condition},cell_line={matched_cell_line}, marker={matched_marker}] marker is None. Skipping it")
                    continue

                file_name = f
                dst_path = self.__get_dst_path(batch, matched_rep, panel, matched_condition, matched_cell_line, matched_marker, file_name)
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

    def get_expected_number_of_files_to_copy(self):
        n = 0
        
        for f in self.get_folders_to_handle():
            f = os.path.join(self.config.SRC_ROOT_PATH, f)
            f_n = os.popen(f"find {f} -type f -name '*{self.config.FILE_EXTENSION}' | wc -l").read()
            n += int(f_n)
            
        return n