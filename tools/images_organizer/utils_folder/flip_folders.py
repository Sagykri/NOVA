import os

def rename_folder(root_folder, target_folder_name, new_folder_name):
    n_folders_renamed = 0
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if target_folder_name in dirnames:
            old_path = os.path.join(dirpath, target_folder_name)
            new_path = os.path.join(dirpath, new_folder_name)
            os.rename(old_path, new_path)
            print(f"Renamed: '{old_path}' to '{new_path}'")
            n_folders_renamed += 1
    print(f"Total number of folders renamed: {n_folders_renamed}")
    
def flip_folders_names(root_folder, folderA, folderB):
    folderA_tmp_name = f"{folderA}_old"
    
    print()
    print(f"{folderA} -----> {folderA_tmp_name}")
    print()
    rename_folder(root_folder, folderA, folderA_tmp_name)
    
    print()
    print(f"{folderB} -----> {folderA}")
    print()
    rename_folder(root_folder, folderB, folderA)
    
    print()
    print(f"{folderA_tmp_name} -----> {folderB}")
    print()
    rename_folder(root_folder, folderA_tmp_name, folderB)
    

root_folder = '/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/raw/Opera18DaysReimaged_sorted/'
folderA_name = 'Tubulin'
folderB_name = 'PSPC1'

flip_folders_names(root_folder, folderA_name, folderB_name)


print("Finished")