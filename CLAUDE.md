# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**NOVA** is a deep learning framework for high-throughput organellar phenotyping of human neurons using Vision Transformers (ViT). It processes microscopy TIFF images → segmented tiles (NPY) → embeddings → UMAPs/distance plots. Paper: [Organellomics](https://www.biorxiv.org/content/10.1101/2024.01.31.572110v1.full).

## Environment

```bash
conda activate nova         # Always use this env for NOVA scripts
export NOVA_HOME=/home/projects/hornsteinlab/giliwo/NOVA
export NOVA_DATA_HOME=...   # Path to data folder (default: $NOVA_HOME/input)
```

## Running Scripts

All runnables take relative config class paths (dot-separated module path from `$NOVA_HOME`):

```bash
# Preprocess raw TIFFs → NPY tiles
python $NOVA_HOME/runnables/preprocessing.py manuscript/dataset_config/MyDatasetConfig

# Train ViT model
python $NOVA_HOME/runnables/train.py manuscript/model_config/ClassificationModelConfig \
    manuscript/trainer_config/ClassificationTrainerConfig \
    manuscript/dataset_config/OpenCellTrainDatasetConfig

# Generate embeddings (needs absolute model path)
python $NOVA_HOME/runnables/generate_embeddings.py $NOVA_HOME/outputs/vit_models/finetuned_model \
    manuscript/embeddings_config/EmbeddingsDatasetConfig

# Generate multiplex embeddings (run after single-marker embeddings)
python $NOVA_HOME/runnables/generate_multiplexed_embeddings.py \
    $NOVA_HOME/outputs/vit_models/finetuned_model manuscript/embeddings_config/EmbeddingsDatasetConfig

# Generate UMAPs
python $NOVA_HOME/runnables/generate_umaps_and_plot.py $NOVA_HOME/outputs/vit_models/finetuned_model \
    manuscript/manuscript_figures_data_config/MyFigureConfig \
    manuscript/manuscript_plot_config/MyPlotConfig

# Calculate distances (optional flags: rep_effect, multiplexed, detailed, normalize)
python $NOVA_HOME/runnables/calculate_distances.py $NOVA_HOME/outputs/vit_models/finetuned_model \
    manuscript/manuscript_figures_data_config/MyFigureConfig multiplexed detailed

# LSF cluster (GPU): use runnables/run.sh with -g flag
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/preprocessing -g -m 20000 -b 10 -j preprocess \
    -a manuscript/dataset_config/MyDatasetConfig
```

## Architecture

### Pipeline Flow
```
Raw TIFFs
  → [preprocessing.py] Cellpose nucleus segmentation + tiling (128→100px, 2-channel)
  → NPY tiles: input/images/processed/{batch}/{cell_line}/{panel}/{condition}/{rep}/{marker}/
  → [train.py] ViT classification training
  → [generate_embeddings.py] Feature extraction → outputs/embeddings/
  → [generate_umaps_and_plot.py] UMAP dimensionality reduction + visualization
  → [calculate_distances.py] Pairwise distance matrices → [plot_distances.py] figures
```

### Directory Structure
```
runnables/      # Entry point scripts (one per pipeline stage)
src/            # Core library
  common/       # BaseConfig, utils (load_config_file, init_logging), log_df
  preprocessing/# Cellpose-based tiling; preprocessor_spd.py / preprocessor_opera.py
  datasets/     # DatasetNOVA: loads NPY tiles, builds labels from folder hierarchy
  models/       # NOVAModel (ViT wrapper), trainers, augmentations
  embeddings/   # generate_embeddings(), save_embeddings()
  analysis/     # Analyzer subclasses: UMAP, distances, effect sizes
  figures/      # Plotting (umap, distances, effect sizes)
  effects/      # EffectsConfig + bootstrap effect size calculation
manuscript/     # Experiment-specific config classes (checked in, versioned)
tools/          # One-off utilities: interactive UMAP, dataset organizers, test scripts
outputs/        # Model checkpoints, embeddings, figures (gitignored)
input/          # Raw and processed images (gitignored)
```

### Configuration System

All parameters live in Python config classes that inherit from `BaseConfig`. Configs are loaded dynamically at runtime by module path string and saved as JSON snapshots in `outputs/configs_used/{timestamp}/` for reproducibility.

Config hierarchy:
```
BaseConfig  (SEED, HOME_FOLDER, OUTPUTS_FOLDER, LOGS_FOLDER)
├── PreprocessingConfig  → src/preprocessing/preprocessing_config.py
├── DatasetConfig        → src/datasets/dataset_config.py
│   ├── EmbeddingsConfig → src/embeddings/embeddings_config.py
│   │   └── EffectConfig → src/effects/effects_config.py
│   └── FiguresConfig    → src/figures/figures_config.py
├── TrainerConfig        → src/trainers/trainer_config.py
└── ModelConfig          → src/models/architectures/model_config.py
└── PlotConfig           → src/figures/plot_config.py
```

**To add a new experiment**: create a new class in `manuscript/` inheriting from the appropriate base config and overriding only the fields that differ.

**Key utility**: `load_config_file(path_string, name, savefolder)` in `src/common/utils.py` — takes a dot-path like `manuscript/my_config/MyClass` and returns a live config instance.

### Data Organization

Input folder hierarchy encodes all metadata (read by `DatasetNOVA` to construct labels):
```
{batch}/{cell_line}/{panel}/{condition}/{rep}/{marker}/filename_s{N}.tiff
```

Processed tiles (NPY shape: `[n_tiles, 100, 100, 2]`) mirror the same hierarchy under `input/images/processed/`.

### Shared sys.path Pattern

Every runnable and many tools insert `NOVA_HOME` into `sys.path`:
```python
sys.path.insert(1, os.getenv("NOVA_HOME"))
```
All cross-module imports use absolute paths from `src.*`.

### Key Classes

| Class | Location | Purpose |
|---|---|---|
| `NOVAModel` | `src/models/architectures/NOVA_model.py` | ViT wrapper; `.load_from_checkpoint()`, `.infer()`, `.generate_embeddings()` |
| `DatasetNOVA` | `src/datasets/dataset_NOVA.py` | PyTorch dataset; builds labels from folder paths |
| `PreprocessorBase` | `src/preprocessing/preprocessors/preprocessor_base.py` | Abstract; Cellpose GPU segmentation, tiling, dead-cell filtering |
| `Analyzer` | `src/analysis/analyzer.py` | Abstract base for all analysis steps; `.calculate()`, `.save()`, `.load()` |
| `BaseConfig` | `src/common/base_config.py` | Root config; property setters trigger initialization side-effects |

### Preprocessing Key Parameters
- Tile size: 100×100px (cropped from 128×128 after segmentation)
- Channels: 2 (target marker + DAPI)
- Max nuclei per tile: 5; min nucleus area inclusion: 80%
- Cellpose: `diameter=70`, `cellprob_threshold=0`, `flow_threshold=0.2`
- Intensity rescaling: percentile bounds [0.5–99.9] (per-channel, skip if `NO_RESCALE_FOR_LOW_SIGNAL_MARKERS`)
