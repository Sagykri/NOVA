# Usage examples:
./bash_commands/run_py.sh ./src/runables/training -g -m 30000 -b 40 -a ./src/models/neuroself/configs/model_config/NeuroselfBatch8BS4TrainingConfig ./src/datasets/configs/train_config/TrainDatasetConfig

./bash_commands/run_py.sh ./src/runables/training -g -m 30000 -b 40 -a ./src/models/cytoself_vanilla/configs/config/CytoselfTrainingConfig ./src/datasets/configs/train_config/TrainOpenCellDatasetConfig

./bash_commands/run_py.sh ./src/runables/preprocessing -g -m 30000 -b 40 -a ./src/preprocessing/configs/spd_batch3/SPD_Batch3  

./bash_commands/run_py.sh ./src/runables/eval_model -g -m 5000 -b 15 -a ./src/models/neuroself/configs/model_config/NeuroselfB78BIT16ShuffleTLTrainingConfig ./src/datasets/configs/test_config/TestB9BIT16DatasetConfig -q gpu-short

./bash_commands/run_py.sh ./src/runables/generate_embeddings -g -m 40000 -b 40 -a ./src/models/neuroself/configs/model_config/NeuroselfB78BIT16ShuffleTLTrainingConfig ./src/datasets/configs/train_config/TrainB78BIT16DatasetConfig 

./bash_commands/run_py.sh ./tests/test_gpu  -m 1000 -g