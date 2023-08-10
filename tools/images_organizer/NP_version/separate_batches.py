import logging
import os
from config_np import DST_ROOT_PATH, SRC_ROOT_PATH, CUT_FILES
import pathlib
from utils import copy_files, get_expected_number_of_files_to_copy, get_folders_to_handle, init_folders, init_logging
import shutil 
from config_np import FILE_EXTENSION

def get_second_batch_wells_index(wells):
    for index, value in enumerate(wells):
        if '6' in str(value):
            return index
    return len(wells)

def get_batches_index_limit(batches_to_wells):
    return 100 * len(batches_to_wells[0])

def main():  
    init_logging()
    # Asserts
    assert os.path.exists(SRC_ROOT_PATH) and os.path.isdir(SRC_ROOT_PATH), f"{SRC_ROOT_PATH} not exists (or not a folder)"
    assert os.path.exists(DST_ROOT_PATH) and os.path.isdir(DST_ROOT_PATH), f"{DST_ROOT_PATH} not exists (or not a folder)"
    
    folders = get_folders_to_handle()
    assert all([os.path.exists(os.path.join(SRC_ROOT_PATH, f)) and os.path.isdir(os.path.join(SRC_ROOT_PATH, f)) for f in folders]), "One or more of the specified folders don't exists (or aren't folders)"
    
    new_out_dir = f"{SRC_ROOT_PATH}_fix"
    pathlib.Path(new_out_dir).mkdir(parents=True, exist_ok=True)
    
    for folder in folders:
        src_path = os.path.join(SRC_ROOT_PATH, folder)
        if 'well' in folder:
            date, batches, panel, wells = folder.split("_")
            batches_list = batches.split("-")
            if len(batches_list)==1:
                dst_path = os.path.join(new_out_dir,folder)
                logging.info(f'copying {src_path} to {dst_path}')
                shutil.copytree(src_path, dst_path)
            else:
                wells_list = wells.split('-')[1:]
                second_batch_indx = get_second_batch_wells_index(wells_list)
                batches_to_wells = {0:wells_list[:second_batch_indx],
                                    1: wells_list[second_batch_indx:]}
                first_batch_index_limit = get_batches_index_limit(batches_to_wells)
                for i, batch in enumerate(batches_list):
                    if 'batch' not in batch:
                        batch = f"batch{batch}"
                    new_batch_wells = 'wells-' + '-'.join(batches_to_wells[i])
                    new_batch_folder = os.path.join(new_out_dir, f'{date}_{batch}_{panel}_{new_batch_wells}')
                    pathlib.Path(new_batch_folder).mkdir(parents=True, exist_ok=True)

                    for file in os.listdir(src_path):
                        file_name,ext = os.path.splitext(file)
                        if ext != FILE_EXTENSION:
                                continue
                        _, info = file_name.split('conf')
                        marker_alias, indx = info.split('_')
                        indx = int(indx.replace('s', ''))
                        if i ==0 and indx > first_batch_index_limit:
                            continue
                        if i==1 and indx <= first_batch_index_limit:
                            continue
                        new_indx=indx
                        if indx > first_batch_index_limit:
                            new_indx = indx - first_batch_index_limit
                        new_file_name = file.replace(f's{str(indx)}', f's{str(new_indx)}')
                        file_dst_path = os.path.join(new_out_dir, new_batch_folder, new_file_name)
                        file_src_path = os.path.join(src_path, file)
                        logging.info(f'old filename: {file}')
                        logging.info(f'new filename: {new_file_name}')
                        logging.info(f'batch{i} copying {file_src_path} to {file_dst_path}')
                        shutil.copy2(file_src_path, file_dst_path)

        else:
            date, batches, panel = folder.split("_")       
            batches_list = batches.split("-")
            if len(batches_list)==1:
                dst_path = os.path.join(new_out_dir,folder)
                logging.info(f'copying {src_path} to {dst_path}')
                shutil.copytree(src_path, dst_path)
            else:
                for i, batch in enumerate(batches_list):
                    if 'batch' not in batch:
                        batch = f"batch{batch}"    
                    new_batch_folder = os.path.join(new_out_dir, f'{date}_{batch}_{panel}')
                    pathlib.Path(new_batch_folder).mkdir(parents=True, exist_ok=True)
                    for file in os.listdir(src_path):
                        file_name,ext = os.path.splitext(file)
                        if ext != FILE_EXTENSION:
                                continue
                        _, info = file_name.split('conf')
                        marker_alias, indx = info.split('_')
                        indx = int(indx.replace('s', ''))
                        if indx >800 and i==0:
                                continue
                        if indx <= 800 and i==1:
                                continue
                        new_indx=indx
                        if indx > 800:
                            new_indx = indx - 800
                        new_file_name = file.replace(f's{str(indx)}', f's{str(new_indx)}')
                        file_dst_path = os.path.join(new_out_dir, new_batch_folder, new_file_name)
                        file_src_path = os.path.join(src_path, file)
                        logging.info(f'old filename: {file}')
                        logging.info(f'new filename: {new_file_name}')
                        logging.info(f'copying {file_src_path} to {file_dst_path}')
                        shutil.copy2(file_src_path, file_dst_path)
                
if __name__ == "__main__":
    main()