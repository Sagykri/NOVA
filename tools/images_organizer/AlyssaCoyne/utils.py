import re
import os
import shutil

from datetime import datetime

from config import Config

class Utils():
    def __init__(self, config:Config):
        self.config = config
    
    
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
            print(f"'FOLDERS' wasn't specified in the config file. Therefore, taking all folders in {self.config.SRC_ROOT_PATH}")
            
            folders = get_subfolders(self.config.SRC_ROOT_PATH)
            
            return get_processed_subfolders(folders)

    def init_folders(self, folder):
        
        # The destination should include these folders
        batch = "batch1" # fixed, since only 1 batch
        panel = "panelA" # fixed, since only 1 panel
        conditions = ['Untreated'] # fixed, since only 1 condition
        
        cell_lines = self.config.CELL_LINES
        markers = self.config.MARKERS
        
        src_folder_path = os.path.join(self.config.SRC_ROOT_PATH, folder, conditions[0])
        print(f"\nSource - Current folder: {src_folder_path}")
            
        self.__create_folders_if_needed(batch, cell_lines)
        
        return src_folder_path, batch, panel, conditions

    def __create_folder_if_needed(self, path):
        if os.path.exists(path) and os.path.isdir(path):
            # Not needed
            return
        print(f"\nCreating folder: {path}")
        os.makedirs(path)
    
    def __create_folders_if_needed(self, batch, cell_lines):
        batch_folder = os.path.join(self.config.DST_ROOT_PATH, batch)
        self.__create_folder_if_needed(batch_folder)
        
        for cell_line in cell_lines:
            cell_line_folder = os.path.join(batch_folder, cell_line)
            self.__create_folder_if_needed(cell_line_folder)
            
    def __get_reps_names(self, folder_path):
        # Nancy & Danino created rep names manuaaly, each subject is a seperate rep
        rep_names = os.listdir(folder_path)
        return rep_names

    def __get_file_info(self, file):
        """
        For example, file named 
        "Control 3_DAPI_TDP-43_Map2_DCP1A_2-Orthogonal Projection-22-Image Export-22_c2.tif"
        The number after "Image Export-" is the site number
        And the "c2" indicates number of channel
        """
        
        # Regular expression to extract site_number and channel_number
        site_match = re.search(r"Image Export-(\d+)", file)
        channel_match = re.search(r"_c([\d-]+)", file)

        # Extract site_number and channel_number
        site_number = site_match.group(1) if site_match else None
        channel_number = channel_match.group(1) if channel_match else None

        return site_number, channel_number
    
    def __get_new_file_name(self, site_number, channel_number):
        """
        For example, file named 
        "Control 3_DAPI_TDP-43_Map2_DCP1A_2-Orthogonal Projection-22-Image Export-22_c2.tif"
        will be changed to 
            "R11_w2confmCherry_s22.tif"
        since the number after "Image Export-" is the site number
        and the "c2" indicates number of channel
        """
        
        mapper = self.config.MARKER_ANTIGEN_NAMES_MAPPER[channel_number]
        return f"R11_w2{mapper['antigen_name']}_s{site_number}.tif"
    
    def copy_files(self, source_folder, destination_folder, cut_files=False,raise_on_missing_index=True):
        n_copied = 0
        
        # This is a rep folder, get all tif files below this folder
        files_names = os.listdir(source_folder)

        for f in files_names:
            src_file_path = os.path.join(source_folder, f)
            file_name,ext = os.path.splitext(f)
            if ext != self.config.FILE_EXTENSION:
                continue
                
            # get the original channel number and the site number
            site_number, channel_number = self.__get_file_info(file_name)
            # prepare the name of the new file (in destination)
            dst_file_name = self.__get_new_file_name(site_number, channel_number)
            
            # prepare the name of the new marker folder
            marker_name = self.config.MARKER_ANTIGEN_NAMES_MAPPER[channel_number]['marker_name']
            self.__create_folder_if_needed(os.path.join(destination_folder, marker_name))
            dst_file_path = os.path.join(destination_folder, marker_name, dst_file_name)

            if cut_files:
                dst_path_full = shutil.move(src_file_path, dst_file_path)
            else:
                dst_path_full = shutil.copy2(src_file_path, dst_file_path)
            n_copied += 1
            print(f"{src_file_path} {'moved' if cut_files else 'copied'} to {dst_path_full}")

        return n_copied

    def get_expected_number_of_files_to_copy(self):
        n = 0
        
        for f in self.get_folders_to_handle():
            f = os.path.join(self.config.SRC_ROOT_PATH, f)
            f_n = os.popen(f"find {f} -type f -name '*{self.config.FILE_EXTENSION}' | wc -l").read()
            n += int(f_n)
            
        return n


    