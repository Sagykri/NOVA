import logging
import os
import shutil
from datetime import datetime
from configCoyne080525 import Config

class Utils:
    """
    A utility class for organizing and copying microscopy image files from a source
    directory to a structured destination path based on filename metadata and configuration.
    """
    def __init__(self, config: Config):
        """
        Initialize the utility class with the given configuration.

        Args:
            config (Config): A Config object holding source/destination paths and mapping details.
        """
        self.config = config

    def init_logging(self, logging_path=None):
        """
        Initialize logging system with both console and file handlers.
        """
        if logging_path is None:
            logging_path = self.config.LOGGING_PATH

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file_name = f"log_{timestamp}.txt"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(os.path.join(logging_path, log_file_name)),
                logging.StreamHandler()
            ]
        )

    def get_folders_to_handle(self):
        """
        Retrieve the folders to be processed based on include/exclude filters.
        """
        # def is_valid(path):
        #     return path not in self.config.EXCLUDE_SUB_FOLDERS and (
        #         not self.config.INCLUDE_SUB_FOLDERS or path in self.config.INCLUDE_SUB_FOLDERS)

        # def list_subfolders(root):
        #     root_path = os.path.join(self.config.SRC_ROOT_PATH, root)
        #     if not os.path.isdir(root_path):
        #         return [] 
        #     return [os.path.join(root, sub) for sub in os.listdir(root_path) if is_valid(os.path.join(root, sub))]

        def list_folders():
            return [f for f in os.listdir(self.config.SRC_ROOT_PATH) if os.path.isdir(os.path.join(self.config.SRC_ROOT_PATH, f))]

        # folders = self.config.FOLDERS or list_subfolders(self.config.SRC_ROOT_PATH)
        # return [sub for folder in folders if os.path.isdir(folder) for sub in os.listdir(folder)]
        
        return self.config.FOLDERS or list_folders()

    def init_folders(self, folder):
        """
        Create the required folder structure under the destination path.
        """
        folder_path = os.path.join(self.config.SRC_ROOT_PATH, folder)
        logging.info(f"Current folder: {folder_path}")

        batch, panel = self.__get_folder_info(folder)
        cell_lines_cfg = self.config.CONFIG[self.config.KEY_CELL_LINES]
        markers = self.config.CONFIG[self.config.KEY_MARKERS][panel]

        for cell_line, data in cell_lines_cfg.items():
            conditions = data.get("conditions", {"Untreated": []})
            reps = data.get("reps", {})

            for rep in reps:
                for condition in conditions:
                    for marker in markers:
                        path = os.path.join(self.config.DST_ROOT_PATH, batch, cell_line, panel, condition, rep, marker)
                        os.makedirs(path, exist_ok=True)

        return folder_path, batch, panel

    def get_expected_number_of_files_to_copy(self):
        """
        Estimate how many files should be processed by scanning source folders.
        """
        total = 0
        for f in self.get_folders_to_handle():
            path = os.path.join(self.config.SRC_ROOT_PATH, f)
            count = os.popen(f"find {path} -type f -name '*c[1-4]{self.config.FILE_EXTENSION}' | wc -l").read()
            total += int(count)
        return total

    def __get_file_info(self, file):
        """
        Parse file name to extract channel, cell line, replicate, and site information.
        """
        name = os.path.splitext(os.path.basename(file))[0]
        parts = name.split("_")
        raw_cell_info = parts[0]
        ch = parts[-1]
        site_str = parts[-2].split("-")[-1]
        site = int(site_str)

        for cell_line, data in self.config.CONFIG.get(self.config.KEY_CELL_LINES, {}).items():
            for rep, alias in data.get("reps", {}).items():
                if raw_cell_info == alias:
                    return ch, cell_line, rep, site, site_str

        raise ValueError(f"Unrecognized cell line alias in: '{raw_cell_info}'")

    def __get_marker_by_alias(self, alias, panel):
        """
        Convert channel alias to marker name.
        """
        aliases = self.config.CONFIG[self.config.KEY_MARKERS_ALIAS_ORDERED]
        if alias not in aliases:
            raise ValueError(f"Unknown marker alias: '{alias}'")
        index = aliases.index(alias)
        return self.config.CONFIG[self.config.KEY_MARKERS][panel][index]

    def __get_dst_path(self, batch, rep, panel, condition, cell_line, marker, file_name):
        """
        Build full destination path for the file.
        """
        return os.path.join(
            self.config.DST_ROOT_PATH, batch, cell_line, panel, condition, rep, marker,
            f"{self.config.FILENAME_POSTFIX}{file_name}"
        )

    def __get_folder_info(self, folder):
        """
        Extract batch and panel names from folder structure.
        """
        batch = os.path.dirname(folder)
        panel = os.path.basename(folder)
        return batch, panel[0].lower() + panel[1:]

    def copy_files(self, folder_path, panel, batch, cut_files=False, raise_on_missing_index=True):
        """
        Copy files to structured destination path based on parsed metadata.
        """
        n_copied = 0
        for f in os.listdir(folder_path):
            logging.info(f"[{os.path.join(folder_path, f)}]")
            file_name, ext = os.path.splitext(f)
            if ext.lower() != self.config.FILE_EXTENSION:
                continue

            try:
                ch, cell_line, rep, site_number, site_number_str = self.__get_file_info(f)
                if ch not in self.config.CONFIG[self.config.KEY_MARKERS_ALIAS_ORDERED]:
                    logging.info(f"Skipping file {f}: invalid channel '{ch}'.")
                    continue

                conditions = self.config.CONFIG.get(self.config.KEY_CELL_LINES, {}).get(cell_line, {}).get("conditions", {"Untreated": []})
                condition = next((c for c, ranges in conditions.items() if any(start <= site_number <= end for start, end in ranges)), "Untreated")

                marker = self.__get_marker_by_alias(ch, panel)
                new_file_name = f"{file_name}_s{site_number_str}{ext}"
                dst_path = self.__get_dst_path(batch, rep, panel, condition, cell_line, marker, new_file_name)
                src_path = os.path.join(folder_path, f)

                if cut_files:
                    shutil.move(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)

                n_copied += 1
                logging.info(f"[{src_path}] {'moved' if cut_files else 'copied'} to {dst_path}")
                print(f"[{src_path}] {'moved' if cut_files else 'copied'} to {dst_path}")

            except Exception as e:
                logging.error(e, exc_info=True)
                if raise_on_missing_index:
                    raise

        return n_copied
