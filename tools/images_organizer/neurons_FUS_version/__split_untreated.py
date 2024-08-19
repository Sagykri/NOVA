import os
import shutil

def copy_files_with_text_in_name(source_dir, dest_dir, search_text):
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        
        # Check if the search text is in the filename
        if search_text in filename:
            # If yes, copy the file to the destination directory
            shutil.copy(file_path, dest_dir)
            print(f"Copied '{file_path}' to '{dest_dir}'")

# source_dir = '/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/processed/spd2/SpinningDisk/FUS_lines_stress_2024_sorted/batch1_Untreated/KOLF/Untreated/ANXA11'
# dest_dir = '/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/processed/spd2/SpinningDisk/FUS_lines_stress_2024_sorted/batch1_Untreated/KOLF/UntreatedBSD/ANXA11'
# search_text = '_BSD_'
# copy_files_with_text_in_name(source_dir, dest_dir, search_text)

# BSD / CCN


source_root_dir = '/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/processed/spd2/SpinningDisk/FUS_lines_stress_2024_sorted/batch1_Untreated/KOLF/Untreated/'
dest_root_dir = '/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/processed/spd2/SpinningDisk/FUS_lines_stress_2024_sorted/batch1_Untreated/KOLF/Untreated_Splitted'
markers = os.listdir(source_root_dir)

for m in markers:
    for tpe in ['BSD', 'CCN', 'MME']:
        source_dir = os.path.join(source_root_dir, m)
        dest_dir = os.path.join(dest_root_dir, f'Untreated_{tpe}', m)
        
        print(f"Source: {source_dir}, Dest: {dest_dir}")
        
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
    
        if tpe == 'BSD' or tpe == 'CCN':
            search_text = f'_{tpe}_'
            copy_files_with_text_in_name(source_dir, dest_dir, search_text)
        else:
            for filename in os.listdir(source_dir):
                file_path = os.path.join(source_dir, filename)
                
                # Check if the search text is in the filename
                if '_BSD_' not in filename and '_CCN_' not in filename:
                    # If yes, copy the file to the destination directory
                    shutil.copy(file_path, dest_dir)
                    print(f"Copied '{file_path}' to '{dest_dir}'")