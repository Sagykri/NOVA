import datetime
import logging
import os
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

def rename_panel_name(directory, src, dst):
    # Loop over each file in the directory
    for filename in os.listdir(directory):
        # Check if the filename starts with src
        if filename.startswith(src):
            # Define the new filename by replacing src with dst
            new_filename = dst + filename[len(src):]
            # Get the full file paths
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(old_file, new_file)
            logging.info(f'Renamed: {old_file} to {new_file}')


def main():
    directory = sys.argv[1]
    src_panel = sys.argv[2] 
    dest_panel = sys.argv[3]
    
    date_str = datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f")
    
    log_file = f"/home/labs/hornsteinlab/Collaboration/MOmaps/tools/images_organizer/neurons_spd18days_version/rename_panel_{date_str}.log"

    init_logging(log_file)
    
    logging.info(f"directory = {directory} src_panel = {src_panel} dest_panel = {dest_panel}")
    
    rename_panel_name(directory, src_panel, dest_panel)
    
    logging.info('Renaming complete.')

if __name__ == "__main__":
    main()
