import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

def get_img(batch_number, cell_line, condition, marker, filename=None, plot=True):
    root_path = "/home/labs/hornsteinlab/Collaboration/MOmaps/input/images/processed/spd2/SpinningDisk"
    subfolder_path = os.path.join(root_path, f"batch{batch_number}",cell_line,condition,marker)
    
    if not os.path.exists(subfolder_path):
        print(f"{subfolder_path} doesn't exists")
        return
    
    if filename is None:
            files = os.listdir(subfolder_path)
            filename = np.random.choice(files)
    file_path = os.path.join(subfolder_path, filename)
    img = np.load(file_path)
    
    if plot:
        n_tiles, n_dim = img.shape[0],img.shape[-1]
        fig,ax = plt.subplots(n_tiles, n_dim)
        for i in range(n_tiles):
            for j in range(n_dim):
                ax[i,j].axis('off')
                ax[i, j].imshow(img[i, ...,j])
        file_id = f"{batch_number}_{cell_line}_{condition}_{marker}_{filename}"
        fig.suptitle(file_id)
        plt.savefig(f"./tests/test_preprocessing/plots/processed_images/{file_id}.png", dpi=3000)
        
    return img

if __name__ == "__main__":
    argv = sys.argv
    if len(argv) > 1:
        batch_number, cell_line, condition, marker = argv[1:5]
        is_filename_supplied = len(argv) > 5
        filename = argv[-1] if is_filename_supplied else None
        
        get_img(batch_number, cell_line, condition, marker, filename, plot=True)
        quit()
    
    """
    NOTE:
    Here you can set the batches, cell lines, conditions and markers to pick from!
    """
    
    batches = [6,7,8,9]
    cell_lines = ["WT", "TDP43", "TBK1"]
    conditions = ["Untreated"]
    markers = ["DAPI", "G3BP1", "PURA", "Phalloidin"]
    # Generate all combinations using meshgrid
    grid = np.meshgrid(batches, cell_lines, conditions, markers, indexing='ij')

    # Concatenate the arrays into a single array
    combs = np.concatenate(grid).reshape(4, -1).T
        
    for comb in combs:
        batch_number, cell_line, condition, marker = tuple(comb)
        get_img(batch_number, cell_line, condition, marker, filename=None, plot=True)
