import logging
import os
import shutil
import datetime
import sys

def init_logging(path):
    """Init logging.
    Writes to log file and console.
    Args:
        path (string): Path to log file
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[
                            logging.FileHandler(path),
                            logging.StreamHandler()
                        ])

def get_overlapping_files(folder1, folder2):
    files1 = [f for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))]
    files2 = [f for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))]

    files1 = set(files1)
    files2 = set(files2)
    
    overlapping_files = files1.intersection(files2)
    return overlapping_files

def create_backup_folder(folder, date_str):
    backup_folder = os.path.join(folder, f"{os.path.basename(folder)}_main_folder_backup_{date_str}")
    logging.info(f"creating backup folder: {backup_folder}")
    os.makedirs(backup_folder, exist_ok=True)
    return backup_folder

def move_files(files, src_folder, dst_folder):
    for file in files:
        src_file_path = os.path.join(src_folder, file)
        dst_file_path = os.path.join(dst_folder, file)
        logging.info(f"[Move] {src_file_path} --> {dst_file_path}")
        shutil.move(src_file_path, dst_file_path)

def move_all_files(src_folder, dst_folder):
    files = os.listdir(src_folder)
    move_files(files, src_folder, dst_folder)

def main():
    main_folder = sys.argv[1] # main folder
    reimaged_folder = sys.argv[2] # reimaged folder
    
    date_str = datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f")
    
    log_file = f"/home/labs/hornsteinlab/Collaboration/MOmaps/tools/images_organizer/neurons_spd18days_version/overlapping_files_{date_str}.log"

    init_logging(log_file)
    
    logging.info(f"Main folder: {main_folder} Reimaged folder: {reimaged_folder}")

    if not os.path.exists(main_folder) or not os.path.isdir(main_folder):
        logging.error(f"The path '{main_folder}' is not valid or is not a directory.")
        return

    if not os.path.exists(reimaged_folder) or not os.path.isdir(reimaged_folder):
        logging.error(f"The path '{reimaged_folder}' is not valid or is not a directory.")
        return

    overlapping_files = get_overlapping_files(main_folder, reimaged_folder)
    logging.info(f"Overlapping files to backup ({len(overlapping_files)}): {list(overlapping_files)}")
    
    if overlapping_files:
        backup_folder = create_backup_folder(main_folder, date_str)
        logging.info(f"Moving overlapping files from main folder ({main_folder}) to backup folder ({backup_folder})")
        move_files(overlapping_files, main_folder, backup_folder)
    
    logging.info(f"Moving all files from reimaged folder ({reimaged_folder}) to main folder ({main_folder})")
    move_all_files(reimaged_folder, main_folder)
    print("Operation completed successfully.")

if __name__ == "__main__":
    main()
