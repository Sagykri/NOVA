# MOmaps

## Prerequisites
0. your conda companion cheat sheet is https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf
1. download anconda 3.7.11: 
	wget https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh: : 
2. follow https://docs.anaconda.com/anaconda/install/linux/. Install the anaconda in Collaboration/MOmaps/anaconda3/

# Environment
3. Create a conda env with the name momaps with python 3.7.11
	conda create --prefix /home/labs/hornsteinlab/Collaboration/MOmaps/anaconda3/momaps python=3.7.11
4. Activate:
conda activate /home/labs/hornsteinlab/Collaboration/MOmaps/anaconda3/momaps

# Packages
5. for each & every package installation do the following:
  a. before: verify that you are in the right env (noted by star) using the following command:
   conda env list
  b. install by: 
   conda install package_name 
  c. After: verify package existance & version (including dependencies) by:
   conda list

6. install the following pakages:
- V opencv-python-headless<4.3 (used pip not conda)
- V cellpose (used pip cellpose[gui])
- V cytoself (used pip not conda)
- V sklearn
- V scikit-image (used pip not conda)
- V umap-learn
- tensorflow (h5py==2.10.0) - For training only:
    V conda install tensorflow-gpu=1.15
    V conda install h5py=2.10.0
- V tqdm
- V shap (shaply values)
- V Shapely (geometric) 
- V adjustText

Add the conda env as kernel to jupyter:
python -m ipykernel install --user --name=momaps

### For Using cellpose with GPU:
- First uninstall the CPU version of torch
  pip uninstall torch
- Then run this command to install the GPU version:
  conda install pytorch cudatoolkit=11.3 -c pytorch

## Runables
- 'preprocessing' - for preprocessing images to be adequate for the model
- 'generate_embeddings' - generate the embedding vectors
- 'training' - for training a model
- 'generate_figures' - for generating the figures for the paper