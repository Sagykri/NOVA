import os
import shutil


batches = [f'batch{i}_16bit_no_downsample' for i in range(2,6)]
cell_lines = [os.path.join('WT','Untreated'),os.path.join('TDP43','Untreated'),os.path.join('TDP43','dox')]
for batch in batches:
    batch_dir = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps','input','images',
                                'processed','spd2','SpinningDisk','deltaNLS',batch)
    for cell_line_folder in cell_lines:
        tdp43_path = os.path.join(batch_dir,cell_line_folder,'TDP43')
        for file in os.listdir(tdp43_path):
            if 'Cy5' in file:
                panel_b_path = os.path.join(batch_dir,cell_line_folder,'TDP43B')
                os.makedirs(panel_b_path, exist_ok=True)
                shutil.copy2(os.path.join(tdp43_path, file), os.path.join(panel_b_path, file))
            elif 'mCherry' in file:
                panel_n_path = os.path.join(batch_dir,cell_line_folder,'TDP43N')
                os.makedirs(panel_n_path, exist_ok=True)
                shutil.copy2(os.path.join(tdp43_path, file), os.path.join(panel_n_path, file))


# batches = [f'batch{i}_16bit_no_downsample' for i in range(2,6)]
# cell_lines = [os.path.join('WT','Untreated'),os.path.join('TDP43','Untreated'),os.path.join('TDP43','dox')]
# for batch in batches:
#     for layer in ['vqvec1','vqvec2']:
#         for cell_line_folder in cell_lines:
#             for rep in ['rep1','rep2']:
#                 path = os.path.join('/home','labs','hornsteinlab','Collaboration','MOmaps_Sagy','MOmaps',
#                 'outputs','models_outputs_batch78_nods_tl_ep23','embeddings','deltaNLS',layer, batch,cell_line_folder, rep,'TDP43')
#                 print(path)
#                 shutil.rmtree(path)
