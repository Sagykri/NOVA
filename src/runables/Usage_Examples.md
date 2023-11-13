# Usage examples:

# Preprocessing of single batch
./bash_commands/run_py.sh ./src/runables/preprocessing -g -m 20000 -b 10 -a ./src/preprocessing/configs/spd_batch3/SPD_Batch3  

# Training neuroself:
./bash_commands/run_py.sh ./src/runables/training -g -m 40000 -b 47 -a ./src/models/neuroself/configs/model_config/TLNeuroselfB78NoDSModelConfig ./src/datasets/configs/training_data_config/B78TrainDatasetConfig -j training_neuroself

# Training cytoself:
./bash_commands/run_py.sh ./src/runables/training -g -m 40000 -b 44  -a ./src/models/cytoself_vanilla/configs/config/CytoselfModelConfig ./src/datasets/configs/training_data_config/TrainOpenCellDatasetConfig -j training_cytoself

# Eval model:
./bash_commands/run_py.sh ./src/runables/eval_model -g -m 10000 -b 25 -a ./src/models/cytoself_vanilla/configs/config/CytoselfModelConfig ./src/datasets/configs/train_config/TrainOpenCellDatasetConfig -q gpu-short -j eval_cytoself

# Generate embeddings:
./bash_commands/run_py.sh ./src/runables/generate_embeddings -g -m 40000 -b 40 -a ./src/models/neuroself/configs/model_config/TLNeuroselfB78NoDSModelConfig ./src/datasets/configs/embeddings_data_config/EmbeddingsB9DatasetConfig -j generate_embeddings_vq2

# Generate vqindhist embeddings (spectral_features)
./bash_commands/run_py.sh ./src/runables/generate_spectral_features -g -m 40000 -b 40 -a ./src/models/neuroself/configs/model_config/TLNeuroselfNiemannPickB14ModelConfig ./src/datasets/configs/embeddings_data_config/EmbeddingsNPB3DatasetConfig -j indhist_NP_3 -q high_gpu_gsla

# Generate UMAP0s
./bash_commands/run_py.sh ./src/runables/generate_umaps -g -m 40000 -b 10 -q high_gpu_gsla -a ./src/models/cytoself_vanilla/configs/config/CytoselfModelConfig ./src/datasets/configs/embeddings_data_config/EmbeddingsU2OSDatasetConfig /home/labs/hornsteinlab/Collaboration/MOmaps/outputs/figures/U2OS

# Generate UMAP2s - Synthetic Multiplexing
./bash_commands/run_py.sh ./src/runables/run_synthetic_multiplexing -g -m 30000 -b 10  -a ./src/models/cytoself_vanilla/configs/config/CytoselfModelConfig ./src/datasets/configs/embeddings_data_config/EmbeddingsU2OSDatasetConfig /home/labs/hornsteinlab/Collaboration/MOmaps/outputs/figures/U2OS -j SM_U2OS

./bash_commands/run_py.sh ./tests/test_gpu  -m 1000 -g