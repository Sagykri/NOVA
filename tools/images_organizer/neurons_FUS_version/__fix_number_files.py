import glob
import os
import shutil
import re

# Copy files and renumbered them

# def copy_and_rename_files(source_dir, dest_dir, value_to_add):
#     # Ensure the destination directory exists
#     if not os.path.exists(dest_dir):
#         print(f"makedirs: {dest_dir}")
#         os.makedirs(dest_dir)

#     # Iterate over all files in the source directory
#     for file_path in glob.glob(os.path.join(source_dir, '*')):
#         filename = os.path.basename(file_path)
#         if not os.path.isfile(os.path.join(source_dir, filename)):
#             print(f"{os.path.join(source_dir,filename)} is not a file")
#             continue
#         # Use regex to find 's' followed by numbers in the filename
#         match = re.search(r's(\d+)', filename)
#         if match:
#             number = int(match.group(1))
#             # Calculate the new number by adding value
#             new_number = number + value_to_add
#             # Replace the original number with the new number in the filename
#             new_filename = re.sub(r's\d+', 's' + str(new_number), filename)
#             # Construct the full destination path for the new file
#             dest_path = os.path.join(dest_dir, new_filename)
#             # Copy the file to the new destination
#             shutil.copy2(file_path, dest_path)
#             print(f"Copied '{file_path}' to '{dest_path}'")

# # Example usage
# source_dir = '/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/FUS_lines_stress_2024_unordered/20240205_Colchicine_Cisplatin_NMS873_2d1d_G-I/PanelG/PanelG_well_B11/'
# dest_dir = '/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/FUS_lines_stress_2024_unordered/20240205_Colchicine_Cisplatin_NMS873_2d1d_G-I/PanelG/PanelG_well_B11/renumbered/'
# value_to_add = 1900
# copy_and_rename_files(source_dir, dest_dir, value_to_add)

# copy renumbered files:
# cp /home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/FUS_lines_stress_2024_unordered/20243001_MG132_ML240_Etoposide_4d_G-I/PanelG/PanelG_wells_B9-11_C10-11/renumbered/* .

# import os
# import shutil

# def copy_files_with_text_in_name(source_dir, dest_dir, search_text):
#     # Ensure the destination directory exists
#     if not os.path.exists(dest_dir):
#         os.makedirs(dest_dir)
    
#     # Iterate over all files in the source directory
#     for filename in os.listdir(source_dir):
#         file_path = os.path.join(source_dir, filename)
        
#         # Check if the search text is in the filename
#         if search_text in filename:
#             # If yes, copy the file to the destination directory
#             shutil.copy(file_path, dest_dir)
#             print(f"Copied '{file_path}' to '{dest_dir}'")

# # Example usage
# # source_dir = '/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/FUS_lines_stress_2024_unordered/20243001_MG132_ML240_Etoposide_4d_A-C/PanelC/panelC_s1900_s2000_well_E11/renumber/'
# # dest_dir = '/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/FUS_lines_stress_2024_sorted/batch/WT/panelC/Untreated/rep1/DAPI/'
# # search_text = 'DAPI'

# source_dir = '/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/'
# dest_dir = '/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/FUS_lines_stress_2024_sorted/batch/WT/panelC/Untreated/rep1/DAPI/'
# search_text = 'DAPI'
# copy_files_with_text_in_name(source_dir, dest_dir, search_text)


# The WT Untreated folder was coppied to KOLF Untreated using the following bash command:
# cp -r /home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/FUS_lines_stress_2024_sorted/batch/WT/panelB/Untreated/ /home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/FUS_lines_stress_2024_sorted/batch/KOLF/panelB/

###############################

# DELETE files in a given range of sites

# import os
# import re

# def delete_files_in_range(directory, start, end):
#     # Compile a regular expression pattern to match 's' followed by a number
#     pattern = re.compile(r's(\d+)')
#     count = 0
#     # Iterate over all files in the given directory
#     for filename in os.listdir(directory):
#         # Use the regular expression to search for 's' followed by numbers in the filename
#         match = pattern.search(filename)
#         if match:
#             # Convert the matched number to an integer
#             number = int(match.group(1))
#             # Check if the number is within the specified range
#             if start <= number <= end:
#                 # Construct the full path to the file
#                 file_path = os.path.join(directory, filename)
#                 # Delete the file
#                 os.remove(file_path)
#                 print(f"Deleted: {file_path}")
#                 count += 1
#     print(f"count = {count}")

# # Example usage
# directory = '/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/FUS_lines_stress_2024_unordered/20240204_Colchicine_Cisplatin_NMS873_2d1d_D-F/PanelE/'
# start_range = 1901
# end_range = 2000
# delete_files_in_range(directory, start_range, end_range)

###############################

# Configure your ranges to letters here
# Example: {(1, 10): 'D', (11, 20): 'E', (21, float('inf')): 'F'}
# panelD_name = 'PanelD_new'
# panelE_name = 'PanelE'
# panelF_name = 'PanelF'

# range_to_letter = {
#     (1,100):panelD_name,
#     (101,200):panelD_name,
#     (201,300):panelE_name,
#     (301,400):panelE_name,
#     (401,500):panelF_name,
#     (501,600):panelF_name,
#     (1101,1200):panelD_name,
#     (1001,1100):panelD_name,
#     (901,1000):panelE_name,
#     (801,900):panelE_name,
#     (701,800):panelF_name,
#     (601,700):panelF_name,
#     (1201,1300):panelD_name,
#     (1301,1400):panelD_name,
#     (1401,1500):panelE_name,
#     (1501,1600):panelE_name,
#     (1601,1700):panelF_name,
#     (1701,1800):panelF_name,
#     (2301,2400):panelD_name,
#     (2201,2300):panelD_name,
#     (2101,2200):panelE_name,
#     (2001,2100):panelE_name,
#     (1901,2000):panelF_name,
#     (1801,1900):panelF_name, # BUG: Was (1801,900):'PanelF'
#     (2401,2500):panelD_name,
#     (2501,2600):panelD_name,
#     (2601,2700):panelE_name,
#     (2701,2800):panelE_name,
#     (2801,2900):panelF_name,
#     (2901,3000):panelF_name,
#     (3001,3100):panelF_name,
#     (3101,3200):panelF_name,
#     (3201,3300):panelE_name,
#     (3301,3400):panelE_name,
#     (3401,3500):panelD_name,
#     (3501,3600):panelD_name,
#     (3601,3700):panelD_name,
#     (3701,3800):panelD_name,
#     (3801,3900):panelE_name,
#     (3901,4000):panelE_name,
#     (4001,4100):panelF_name,
#     (4101,4200):panelF_name,
#     (4201,4300):panelF_name,
#     (4301,4400):panelF_name,
#     (4401,4500):panelE_name,
#     (4501,4600):panelE_name,
#     (4601,4700):panelD_name,
#     (4701,4800):panelD_name,
#     (4801,4900):panelD_name,
#     (4901,5000):panelD_name,
#     (5001,5100):panelE_name,
#     (5101,5200):panelE_name,
#     (5201,5300):panelF_name,
#     (5301,5400):panelF_name,
#     (5401,5500):panelF_name,
#     (5501,5600):panelF_name,
#     (5601,5700):panelE_name,
#     (5701,5800):panelE_name,
#     (5801,5900):panelD_name,
#     (5901,6000):panelD_name
# }

# def map_number_to_letter(number):
#     for (start, end), letter in range_to_letter.items():
#         if start <= number <= end:
#             return letter
#     return None

# def copy_files_to_folders(source_sub_folder):
#     # Regular expression to match files ending with 's' followed by numbers
#     file_pattern = re.compile(r's(\d+).tif$')
    
#     for filename in os.listdir(source_sub_folder):
#         match = file_pattern.search(filename)
#         if match:
#             number = int(match.group(1))
#             letter = map_number_to_letter(number)
#             if letter:
#                 # Create the target folder if it doesn't exist
#                 source_folder = os.path.dirname(source_sub_folder)
#                 target_folder = os.path.join(source_folder, letter)
                
#                 os.makedirs(target_folder, exist_ok=True)
                
#                 source_path = os.path.join(source_sub_folder, filename)
#                 target_path = os.path.join(target_folder, filename)
#                 # # Copy the file
#                 # shutil.copy2(source_path, target_path)
#                 # print(f"Copied '{filename}' to '{target_folder}'")
                
#                 # MOVE the file!!!!!!!
#                 #shutil.move(source_path, target_path)
#                 #print(f"'{filename}' moved to '{target_folder}'")

# # Example usage
# # source_sub_folder = '/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/FUS_lines_stress_2024_unordered/20242801_BMAA_SA_DMSO_4d_D-F/PanelDEF'  # Replace with the path to your source folder
# source_sub_folder = "/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/SpinningDisk/FUS_lines_stress_2024_unordered/20242801_BMAA_SA_DMSO_4d_D-F/PanelF/"
# copy_files_to_folders(source_sub_folder)

# # I fixed the numbering of D and F, then ran this script when source_sub_folder once was PanelD and the panelF_name was 'PanelF_new', and another time when source_sub_folder was PanelF and panelD_name was 'PanelD_new'
# # Then, I copied the content of PanelD_new to PanelD and the same for PanelF+PanelF_new