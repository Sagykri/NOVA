import datetime
import logging
import os
import shutil
import re
import glob
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

# Copy files and renumbered them
def copy_and_rename_files(source_dir, dest_dir, value_to_add):
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        logging.info(f"makedirs: {dest_dir}")
        os.makedirs(dest_dir)

    # Iterate over all files in the source directory
    for file_path in glob.glob(os.path.join(source_dir, '*')):
        filename = os.path.basename(file_path)
        if not os.path.isfile(os.path.join(source_dir, filename)):
            logging.warning(f"{os.path.join(source_dir,filename)} is not a file")
            continue
        # Use regex to find 's' followed by numbers in the filename
        match = re.search(r's(\d+)', filename)
        if match:
            number = int(match.group(1))
            # Calculate the new number by adding value
            new_number = number + value_to_add
            # Replace the original number with the new number in the filename
            new_filename = re.sub(r's\d+', 's' + str(new_number), filename)
            # Construct the full destination path for the new file
            dest_path = os.path.join(dest_dir, new_filename)
            # Copy the file to the new destination
            shutil.copy2(file_path, dest_path)
            logging.info(f"Copied '{file_path}' to '{dest_path}'")

def main():
    # Example usage
    # source_dir = '/home/projects/hornsteinlab/Collaboration/MOmaps/input/images/raw/SpinningDisk/NOVA_d18_neurons/batch2/PanelB/well D11/'
    # dest_dir = '/home/projects/hornsteinlab/Collaboration/MOmaps/input/images/raw/SpinningDisk/NOVA_d18_neurons/batch2/PanelB/well D11/renumbered/'
    # value_to_add = 1900
    source_dir = sys.argv[1] # main folder
    dest_dir = sys.argv[2] # reimaged folder
    value_to_add = int(sys.argv[3])
    
    date_str = datetime.datetime.now().strftime("%d%m%y_%H%M%S_%f")
    
    log_file = f"/home/projects/hornsteinlab/Collaboration/MOmaps/tools/images_organizer/neurons_spd18days_version/renumbering_{date_str}.log"

    init_logging(log_file)
    
    logging.info(f"source_dir = {source_dir} dest_dir = {dest_dir} value_to_add = {value_to_add}")
    
    copy_and_rename_files(source_dir, dest_dir, value_to_add)

if __name__ == "__main__":
    main()
