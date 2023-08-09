
███╗░░░███╗░█████╗░███╗░░░███╗░█████╗░██████╗░░██████╗
████╗░████║██╔══██╗████╗░████║██╔══██╗██╔══██╗██╔════╝
██╔████╔██║██║░░██║██╔████╔██║███████║██████╔╝╚█████╗░
██║╚██╔╝██║██║░░██║██║╚██╔╝██║██╔══██║██╔═══╝░░╚═══██╗
██║░╚═╝░██║╚█████╔╝██║░╚═╝░██║██║░░██║██║░░░░░██████╔╝
╚═╝░░░░░╚═╝░╚════╝░╚═╝░░░░░╚═╝╚═╝░░╚═╝╚═╝░░░░░╚═════╝░

## Prerequisites

# NANCY: cellpose needs python 3.8 (and also cell profiler), so installed 
1. Go to MOmaps main folder, and run this to download the .sh file:
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh

 your conda companion cheat sheet is https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

2. Install Anaconda under the shared folder
	Follow https://docs.anaconda.com/anaconda/install/linux/
	Install the anaconda in specific path: Collaboration/MOmaps/anaconda3/

# Environment
3. Create a conda env with the name momaps with python 3.8
	conda create --prefix /home/labs/hornsteinlab/Collaboration/MOmaps/anaconda3/momaps python=3.8
4. Activate:
	conda activate /home/labs/hornsteinlab/Collaboration/MOmaps/anaconda3/momaps

# Packages
5. For each & every package installation do the following:
  a. before: verify that you are in the right env (noted by star) using the following command:
   conda env list
  b. install by: 
   conda install package_name 
  c. After: verify package existance & version (including dependencies) by:
   conda list
  d. To use pip, to:
	/home/labs/hornsteinlab/Collaboration/MOmaps/anaconda3/momaps/bin/python -m pip install package_name
6. install the following pakages:

- Set the pip to work with the installed conda:
	/home/labs/hornsteinlab/Collaboration/MOmaps/anaconda3/momaps/bin/python -m pip install --user --upgrade pip

### For Using cellpose with GPU:

- First uninstall the CPU version of torch
  pip uninstall torch
- Then run this command to install the GPU version:
	new: conda install pytorch=1.12.1 cudatoolkit=11.3.1 -c pytorch -c conda-forge
  old: conda install pytorch cudatoolkit=11.4 -c pytorch -c conda-forge
  Notice (4.5.23): Unfortunately, with cudatoolkit 11.4, conda only proposes the CPU version of pytorch.
  You should install cudatoolkit=11.3.1 and pytorch=1.12.1 for the GPU version!

- install cellpose:
	/home/labs/hornsteinlab/Collaboration/MOmaps/anaconda3/momaps/bin/python -m pip install cellpose==2.0

- /home/labs/hornsteinlab/Collaboration/MOmaps/anaconda3/momaps/bin/python -m pip install "opencv-python-headless<4.3" 

- /home/labs/hornsteinlab/Collaboration/MOmaps/anaconda3/momaps/bin/python -m pip install "opencv-python-headless<4.3" install cytoself 

- conda install scikit-learn

- conda install numpy

- conda install -c anaconda pandas

- conda install scikit-image

- conda install -c conda-forge umap-learn
 
- tensorflow (h5py==2.10.0) - For training only:
    conda install tensorflow-gpu=1.15
    conda install h5py=2.10.0

- conda install -c conda-forge tqdm matplotlib shap shapely adjusttext

At the end, recheck that the pytorch you have installed is indeed the GPU version and wasn't switched to the CPU version!
(If GPU isn't available all of a sudden, try to uninstall numpy)


Add the conda env as kernel to jupyter:

  /home/labs/hornsteinlab/Collaboration/MOmaps/anaconda3/momaps/bin/python -m pip install ipykernel --user

  /home/labs/hornsteinlab/Collaboration/MOmaps/anaconda3/momaps/bin/python -m ipykernel install --user --name momaps --display-name momaps


## Runables
- 'preprocessing' - for preprocessing images to be adequate for the model
- 'generate_embeddings' - generate the embedding vectors
- 'training' - for training a model
- 'generate_figures' - for generating the figures for the paper

## Pay Attention!
In order to run anything here, you must set an environment variable "MOMAPS_HOME" as your MOmaps home directory.
```
export MOMAPS_HOME=PATH_TO_HOME_FOLDER
```
Put the above inside the ~/.bash_profile file in order to set the variable automatically whenever you login

If your data folder is in a different location than MOMAPS_HOME/input, please define MOMAPS_DATA_HOME to be the path to your 'input' folder.
```
export MOMAPS_DATA_HOME=PATH_TO_INPUT_FOLDER
```

Also, add:
```
module load cuda/11.7
```
to your  /home/labs/hornsteinlab/your_user_name/.bashrc file
