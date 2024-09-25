# NOVA

![Python: 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)
![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)

Welcome to the official repository for **NOVA**, a deep learning framework designed for high-throughput organellar phenotyping of human neurons using AI-driven methodologies. This repository contains the code, data, and instructions needed to reproduce the results described in our paper: [Organellomics: AI-driven Deep Organellar Phenotyping of Human Neurons](https://www.biorxiv.org/content/10.1101/2024.01.31.572110v1.full).

## Table of Contents

- [NOVA](#nova)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Download the model](#download-the-model)
  - [Usage](#usage)
    - [Generate Embeddings](#generate-embeddings)
    - [Generate UMAPs](#generate-umaps)
    - [Generate distances plots](#generate-distances-plots)
      - [Calculate distances](#calculate-distances)
      - [Generate plots](#generate-plots)
        - [Boxplots](#boxplots)
        - [Bubbleplots](#bubbleplots)
    - [Train a model](#train-a-model)
  - [Data](#data)
  - [Configuration files](#configuration-files)
      - [Base Config](#base-config)
      - [Preprocessing Config](#preprocessing-config)
      - [Dataset Config](#dataset-config)
      - [Trainer Config](#trainer-config)
      - [Model Config](#model-config)
      - [Embeddings Config](#embeddings-config)
      - [Figures Config](#figures-config)
      - [Plot Config](#plot-config)
    - [Creating a new configuration](#creating-a-new-configuration)

## Introduction

**NOVA** is an AI-driven deep learning approach for analyzing and phenotyping organelles in human neurons at scale. The model leverages state-of-the-art neural networks to identify and classify various organelles in microscopy images, enabling new insights into cellular biology and potential implications for neurodegenerative diseases.

## Installation

To get started with Organellomics, clone the repository and install the required dependencies.

```bash
git clone https://github.com/Sagykri/NOVA.git
cd NOVA
conda env create --name nova --file environment_nova.yml
```

Next, you need to set two environment variables:
```bash
export NOVA_HOME=*path to NOVAs root folder*
export NOVA_DATA_HOME=*path to the data folder*
```

## Download the model
The model file can be downloaded from here:
[model.pth](https://github.com/Sagykri/MOmaps/tree/main/outputs/vit_models/finetuned_model/model.pth) 

The pretrained model also be downloaded using the link: [pretrained_model.pth](https://github.com/Sagykri/MOmaps/tree/main/outputs/vit_models/pretrained_model/model.pth)

## Usage

### Preprocess the data
```bash
python $NOVA_HOME/runnables/preprocessing *RELATIVE_PATH_TO_DATASET_CONFIG_CLASS*
```

For example:
```bash
python $NOVA_HOME/runnables/preprocessing /manuscript/dataset_config/OpenCellTrainDatasetConfig
```

For WEXAC:
```bash
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/preprocessing -g -m 20000 -b 10 -j preprocess -a ./manuscript/dataset_config/OpenCellTrainDatasetConfig
```

### Train a model
```bash
python $NOVA_HOME/runnables/train *RELATIVE_PATH_TO_MODEL_CONFIG_CLASS* *RELATIVE_PATH_TO_TRAINER_CONFIG_CLASS* *RELATIVE_PATH_TO_DATASET_CONFIG_CLASS*
```

For example:
```bash
python $NOVA_HOME/runnables/train ./manuscript/model_config/ClassificationModelConfig /manuscript/trainer_config/ClassificationTrainerConfig  /manuscript/dataset_config/OpenCellTrainDatasetConfig
```

For WEXAC:
```bash
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/train -g -m 40000 -b 44 -j train -a ./manuscript/model_config/ClassificationModelConfig ./manuscript/trainer_config/ClassificationTrainerConfig  ./manuscript/dataset_config/OpenCellTrainDatasetConfig
```

<br/>

### Generate Embeddings
Once you have a trained model, you may proceed to generate embeddings for further analysis and plots creation.
To generate embeddings you need to run the following:

```bash
python $NOVA_HOME/runnables/generate_embeddings *ABSOLUTE_PATH_TO_MODEL_FOLDER* *RELATIVE_PATH_TO_EMBEDDINGS_CONFIG_CLASS*
```

For example:
```bash
python $NOVA_HOME/runnables/generate_embeddings $NOVA_HOME/outputs/vit_models/finetuned_model ./manuscript/embeddings_config/EmbeddingsDatasetConfig
```

On WEXAC:
```bash
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 -a $NOVA_HOME/outputs/vit_models/finetuned_model ./manuscript/embeddings_config/AlyssaEmbeddingsDatasetConfig -q short-gpu -j generate_embeddings
```
<br/>

### Generate UMAPs
Once you have saved embeddngs, you may generate UMAPs for them following this command:

```bash
python $NOVA_HOME/runnables/generate_umaps_and_plot *ABSOLUTE_PATH_TO_MODEL_FOLDER* *RELATIVE_PATH_TO_FIGURES_CONFIG_CLASS* *RELATIVE_PATH_TO_PLOT_CONFIG_CLASS*
```

For example:
```bash
python $NOVA_HOME/runnables/generate_umaps_and_plot $NOVA_HOME/vit_models/finetuned_model ./manuscript/manuscript_figures_data_config/NeuronsUMAP0StressB6FigureConfig ./manuscript/manuscript_plot_config/UMAP0StressPlotConfig
```

On WEXAC:
```bash
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 5000 -a  $NOVA_HOME/vit_models/finetuned_model ./manuscript/manuscript_figures_data_config/NeuronsUMAP0StressB6FigureConfig ./manuscript/manuscript_plot_config/UMAP0StressPlotConfig -q short -j generate_umap
```
<br/>

### Generate distances plots
For generating distances plots you should first calculate the distances.

#### Calculate distances
For calculating distance you should run the following:
```bash
python $NOVA_HOME/runnables/generate_distances *ABSOLUTE_PATH_TO_MODEL_FOLDER* *RELATIVE_PATH_TO_DISTANCE_CONFIG_CLASS*
```
For example:
```bash
python $NOVA_HOME/runnables/generate_distances $NOVA_HOME/outputs/vit_models/finetuned_model ./manuscript/distances_config/DistanceConfig 
```
On WEXAC:
```bash
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_distances -a /home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/finetuned_model ./manuscript/distances_config/NeuronsDistanceConfig -j dist_neurons
```
<br/>

#### Generate plots
Once you have the distances calculated, you may plot them with the following plot types:

##### Boxplots
```bash
python $NOVA_HOME/runnables/plot_distances_boxplots *ABSOLUTE_PATH_TO_MODEL_FOLDER* *RELATIVE_PATH_TO_FIGURES_CONFIG_CLASS* *RELATIVE_PATH_TO_PLOT_CONFIG_CLASS*
```

For example:
```bash
python $NOVA_HOME/runnables/plot_distances_boxplots $NOVA_HOME/outputs/vit_models/finetuned_model ./manuscript/manuscript_figures_data_config/DistancesFigureConfig ./manuscript/manuscript_plot_config/DistancesPlotConfig
```

For WEXAC:
```bash
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_distances_boxplots -m 1000 -a $NOVA_HOME/outputs/vit_models/finetuned_model ./manuscript/manuscript_figures_data_config/DistancesFigureConfig ./manuscript//manuscript_plot_config/DistancesPlotConfig -q short -j boxplots
```

##### Bubbleplots
```bash
python $NOVA_HOME/runnables/plot_distances_bubble *ABSOLUTE_PATH_TO_MODEL_FOLDER* *RELATIVE_PATH_TO_FIGURES_CONFIG_CLASS* *RELATIVE_PATH_TO_PLOT_CONFIG_CLASS*
```

For example:
```bash
python $NOVA_HOME/runnables/plot_distances_bubble $NOVA_HOME/outputs/vit_models/finetuned_model ./manuscript/manuscript_figures_data_config/DistancesFigureConfig ./manuscript/manuscript_plot_config/DistancesPlotConfig
```

For WEXAC:
```bash
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_distances_bubble -m 1000 -a $NOVA_HOME/outputs/vit_models/finetuned_model ./manuscript/manuscript_figures_data_config/DistancesFigureConfig ./manuscript/manuscript_plot_config/DistancesPlotConfig -q short -j boxplots
```



## Data
Your data folder should be organized as follows:
```
- The first batch’s name and its number (for example batch1)
  - The name of the first cell line (for example, WT, TDP43, ...)
    - panelA
      - The name of the first condition (for example, Untreated, stress, ..)
        - rep1
          - The name of the first marker (for example DAPI, G3BP1, ..)
              - The name of the first site with some site index (preferably ‘_s1’) and a ‘.tiff’ extension
                - The name of the second site with some site index (preferably ‘_s2’) and a ‘.tiff’ extension ...
            - The name of the second marker
            - …
        - rep2
        - …
      - The name of the second condition
      - …
    - panelB
    - panelC
    - …
  - The name of the second cell line
  - …
- The second batch’s name and its number (for example batch2)
- …
```

For example:
```
- batch1
  - WT
    - panelA
      - Untreated
        - rep1
          - DAPI
            - filename_s1.tiff
            - filename_s2.tiff
          - G3BP1
            - filename_s1.tiff
            - filename_s2.tiff
          - ...
        - rep2
          - DAPI
            - filename_s101.tiff
            - filename_s102.tiff
          - G3BP1 
            - filename_s101.tiff
            - filename_s102.tiff
          - ...
      - stress
        - rep1
          - DAPI
            - filename_s201.tiff
            - filename_s202.tiff
          - ...
      - ...
    - panelB
      - Untreated
        - rep1
          - DAPI
            - filename_s401.tiff
            - filename_s402.tiff
          - ...
    - ...
  - TDP43
    - panelA
      - Untreated
        - rep1
          - DAPI
            - filename_s601.tiff
            - filename_s602.tiff
  - ...
```

<h3><u>Note:</u> <br/></h3>
Make sure you folders names **don't** include **underscores** ('_').<br/>
You may use **CamelCase** instead.

## Configuration files

#### Base Config
```python
# The seed
SEED:int

# The path to the root folder of the project
HOME_FOLDER:str

# The path to the root input folder 
HOME_DATA_FOLDER:str

# The path where to save the configuration files that have been used
CONFIGS_USED_FOLDER:str

# The path to the root logs folder
LOGS_FOLDER:str
```

#### Preprocessing Config
```python
# The path to the raw data folder
RAW_FOLDER_ROOT:str 

# The path to the output (processed) folder
PROCESSED_FOLDER_ROOT:str 

# An array of all the folders to process
INPUT_FOLDERS:List[str] 

# An array to where to save the processed files
PROCESSED_FOLDERS:List[str] 

# The expected image shape
EXPECTED_IMAGE_SHAPE:Tuple[int, int] 

# The tile shape when cropping the image into tiles
TILE_INTERMEDIATE_SHAPE:Tuple[int,int] 

# The final tile shape after resizing from TILE_INTERMEDIATE_SHAPE
TILE_SHAPE:Tuple[int, int] 

# Maximum allowed nuclei in a tile
MAX_NUM_NUCLEI:int

# Num of workers to use when running the preprocessing in parallel
NUM_WORKERS:int 

# Settings for cellpose 
# For more details please see: https://cellpose.readthedocs.io/en/latest/settings.html
CELLPOSE = {
            'NUCLEUS_DIAMETER': int,
            'CELLPROB_THRESHOLD': int,
            'FLOW_THRESHOLD': float
        } 

# The lower and upper bounds *percentiles* to shrink the image intenstiy into
# Requirement: 0<=lower_bound<=upper_bound<=100
# For more details see: https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.rescale_intensity 
RESCALE_INTENSITY = {
            'LOWER_BOUND': float,
            'UPPER_BOUND': float
        }  

# The path to the file holding the focus boundries for each marker
MARKERS_FOCUS_BOUNDRIES_PATH:Union[None,str] 

# Which markers to include
MARKERS:Union[None, List[List]]

# Cell lines to include
CELL_LINES:Union[None, List[List]]

# Conditions to include
CONDITIONS:Union[None, List[List]]

# Reps to include
REPS:Union[None, List[List]]  

# The path to the Preprocessor class (the path to the py file, then / and then the name of the class)
# ex: os.path.join("src", "preprocessing", "preprocessor_spd", "SPDPreprocessor")
PREPROCESSOR_CLASS_PATH:str

```

#### Dataset Config
```python
# The path to the root of the processed folder
PROCESSED_FOLDER_ROOT:str

# The path to the data folders
INPUT_FOLDERS:List[str]

# Which markers to include
MARKERS:List[str]

# Which markers to exclude
MARKERS_TO_EXCLUDE:List[str]

# Cell lines to include
CELL_LINES:List[str]

# Conditions to include
CONDITIONS:List[str]

# Reps to include
REPS:List[str]

# Should split the data to train,val,test?
SPLIT_DATA:bool

# The percentage of the data that goes to the training set
TRAIN_PCT:float

# Should shuffle the data within each batch collected?
##Must be true whenever using SPLIT_DATA=True otherwise train,val,test set won't be the same as when shuffle was true
SHUFFLE:bool   

# Should add the cell line to the label?
ADD_LINE_TO_LABEL:bool

# Should add condition to the label?
ADD_CONDITION_TO_LABEL:bool 

# Should add the batch to the label?
ADD_BATCH_TO_LABEL:bool

# Should add the rep to the label?
ADD_REP_TO_LABEL:bool

# Number of channels per image
NUM_CHANNELS:int

# The size of each image (width,height)
IMAGE_SIZE:Tuple[int, int]
```

#### Trainer Config
```python
# The starting learning rate
LR:float

# The final learning rate at the end of the schedule
MIN_LR:float

# Number of epochs
MAX_EPOCHS:int

# Number of epochs to warmup the learning rate
WARMUP_EPOCHS:int

# The starting weight decay value
WEIGHT_DECAY:float

# The final weight decay value at the end of the schedule
WEIGHT_DECAY_END:float

# The batchsize (how many files to load per batch)
BATCH_SIZE:int

# Number of works to run during the data loading
NUM_WORKERS:int

# Number of straight epochs without improvement to wait before activating eary stopping 
EARLY_STOPPING_PATIENCE:int

# The path to the trainer class (the path to the py file, then / and then the name of the class)
# ex: os.path.join("src", "common", "lib", "models", "trainers", "trainer_classification", "TrainerClassification")
TRAINER_CLASS_PATH:str

# A textual description for the model (optional, default to the trainer class name)
DESCRIPTION:str

# Whether to drop the last batch if its partial of the expected batch size
DROP_LAST_BATCH:bool

# The path to the aumentation to apply on each sample in the data (the path to the py file, then / and then the name of the class)
# ex: os.path.join("src", "models", "utils", "augmentations", "RotationsAndFlipsAugmentation")
DATA_AUGMENTATION_CLASS_PATH:str
```

#### Model Config
```python
# The version of the vit (options: 'tiny'|'small'|'base')
VIT_VERSION:str

# The image size (weight==height) the model would expect
IMAGE_SIZE:int

# The patch size for the model
PATCH_SIZE:int

# Num of channels the model would expect in the input sampels
NUM_CHANNELS:int

# The size of the model's output 
OUTPUT_DIM:int
```

#### Embeddings Config
```python
# The path to the data folders
INPUT_FOLDERS:List[str]

# The name for the experiment
EXPERIMENT_TYPE:str

# Which dataset type to load (options: 'trainset', 'valset', 'testset')
SETS:List[str]
```

#### Figures Config
```python
# The path to the data folders
INPUT_FOLDERS:List[str]

# Decide if to show ARI metric on the UMAP
SHOW_ARI:bool
```

#### Plot Config
```python
# Set the size of the dots
SIZE:int

# Set the alpha of the dots (0=max opacity, 1=no opacity)
ALPHA:float 

# Set the color mapping dictionary (name: {alias:alias, color:color})
COLOR_MAPPINGS:Dict[str, Dict[str,str]]
```

### Creating a new configuration

For creating a new configuration you should create (or use an existsing) python file, set there a class with a representative name and make it inherit from the relevant class (BaseConfig/PreprocessingConfig/ModelConfig etc..).

<u>For example:</u>

New dataset configuration:
```python
class OpenCellDatasetConfig(DatasetConfig):
    def __init__(self):
        super().__init__()

        self.INPUT_FOLDERS = [os.path.join(self.PROCESSED_FOLDER_ROOT, "OpenCell")]

        self.SPLIT_DATA = True
        self.MARKERS_TO_EXCLUDE = ['DAPI']

        ######################
        self.CELL_LINES = ['WT']
        self.CONDITIONS = ['Untreated']
        ######################
```
New model configuration:
```python
class ClassificationModelConfig(ModelConfig):
    """Configuration for the classification model
    """
    def __init__(self):
        super().__init__()
        
        self.OUTPUT_DIM = 1311
```
