
import os
import sys
sys.path.insert(1, os.getenv("MOMAPS_HOME"))

import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf as bpdf
from src.common.lib.preprocessing_utils import rescale_intensity


OUTLIERS_PATH = "../MOmaps_Sagy/MOmaps/sandbox/outliers_detection/outliers_paths_from_umap/"
OUTPUT_PATH = os.path.join(OUTLIERS_PATH, "4Lena")

def do():
    
    folders = os.listdir(OUTLIERS_PATH)
    folders.remove('batch6_rep2_WT_untreated_stress')
    folders.remove('4Lena')
    for batch_rep in folders:
        batch_rep_path = os.path.join(OUTLIERS_PATH, batch_rep)
        cell_lines = os.listdir(batch_rep_path)
        
        for cell_line in cell_lines:
            cell_line_path = os.path.join(OUTLIERS_PATH, batch_rep, cell_line)
            marker_files = os.listdir(cell_line_path)
            
            for marker_file in marker_files:
                # PFD file name for each cell line and marker
                file_name = f"{cell_line}_{marker_file.replace('.npy','')}_{batch_rep}_outlier_tiles_4Lena.pdf"
                pdf_file = os.path.join(OUTPUT_PATH, file_name)
                pdf_pages = bpdf.PdfPages(pdf_file)
                print(f"\ncreate a pdf file: {pdf_file}")
                
                # Load list of outliers (paths to preprocessed npy) 
                marker_outliers_paths = np.load(os.path.join(cell_line_path, marker_file))
                for image_path in marker_outliers_paths:
                    
                    f, ax = plt.subplots(1,2, figsize=(3,3))
                    
                    path_split = image_path.split(".npy_")
                    preprocessed_image_path, tile_num = path_split[0]+".npy", int(path_split[-1])
                    # Load the image tiles
                    img = np.load(preprocessed_image_path)
                    
                    # plot target image
                    ax[0].text(0,0,image_path.split("/")[-1],fontsize=4,color="red")
                    ax[0].imshow(img[tile_num,:,:,0], cmap='gray', vmin=0, vmax=1)
                    ax[0].axis("off")
                    
                    # DAPI image
                    ax[1].imshow(img[tile_num,:,:,1], cmap='gray', vmin=0, vmax=1)
                    ax[1].axis("off")
                    pdf_pages.savefig()
                    
                pdf_pages.close()
                
if __name__ == "__main__":

    do()