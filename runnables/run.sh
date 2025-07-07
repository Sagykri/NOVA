#!/bin/bash

# Usage example:

# Test GPU
# ./bash_commands/run_py.sh ./tests/test_gpu  -m 1000 -g

# Preprocessing of single batch
# ./bash_commands/run_py.sh ./src/runables/preprocessing -g -m 20000 -b 10 -a ./src/preprocessing/configs/spd_batch3/SPD_Batch3 True  -j preprocessing_SPD_Batch3

# Training neuroself:
# ./bash_commands/run_py.sh ./src/runables/training -g -m 40000 -b 47 -a ./src/models/neuroself/configs/model_config/TLNeuroselfB78NoDSModelConfig ./src/datasets/configs/training_data_config/B78TrainDatasetConfig -j training_neuroself

# Training cytoself:
# ./bash_commands/run_py.sh ./src/runables/training -g -m 40000 -b 44  -a ./src/models/cytoself_vanilla/configs/config/CytoselfModelConfig ./src/datasets/configs/training_data_config/TrainOpenCellDatasetConfig -j training_cytoself -q gsla_high_gpu

# Eval model:
# ./bash_commands/run_py.sh ./src/runables/eval_model -g -m 10000 -b 25 -a ./src/models/cytoself_vanilla/configs/config/CytoselfModelConfig ./src/datasets/configs/train_config/TrainOpenCellDatasetConfig -q gpu-short -j eval_cytoself

# Generate embeddings (vqvec2):
# ./bash_commands/run_py.sh ./src/runables/generate_embeddings -g -m 40000 -b 40 -a ./src/models/neuroself/configs/model_config/TLNeuroselfB78NoDSModelConfig ./src/datasets/configs/embeddings_data_config/EmbeddingsB9DatasetConfig -j generate_embeddings_vq2

# Generate vqindhist embeddings (spectral_features)
# ./bash_commands/run_py.sh ./src/runables/generate_spectral_features -g -m 40000 -b 40 -a ./src/models/neuroself/configs/model_config/TLNeuroselfB78NoDSModelConfig ./src/datasets/configs/embeddings_data_config/EmbeddingsNPB3DatasetConfig -j indhist_NP_3 -q gsla_high_gpu

# Generate UMAP0s
# ./bash_commands/run_py.sh ./src/runables/generate_umaps -g -m 40000 -b 10 -a ./src/models/neuroself/configs/model_config/TLNeuroselfB78NoDSModelConfig ./src/datasets/configs/embeddings_data_config/EmbeddingsB6DatasetConfig /home/projects/hornsteinlab/Collaboration/MOmaps/outputs/figures/manuscript/fig2/neuroself/ -j gen_umaps_b6_stress
# ./bash_commands/run_py.sh ./src/runables/generate_umaps -g -m 40000 -b 10 -q high_gpu_gsla -a ./src/models/cytoself_vanilla/configs/config/CytoselfModelConfig ./src/datasets/configs/embeddings_data_config/EmbeddingsU2OSDatasetConfig /home/projects/hornsteinlab/Collaboration/MOmaps/outputs/figures/U2OS -j gen_umaps

# Generate UMAPS0 from vqindhist
# ./bash_commands/run_py.sh ./src/runables/generate_umaps0_vqindhist -g -m 50000 -b 1 -a ./src/models/neuroself/configs/model_config/TLNeuroselfB78NoDSModelConfig ./src/figures/figures_config/OperaUMAP0_B1FigureConfig /home/projects/hornsteinlab/Collaboration/MOmaps/outputs/figures/Opera_18days/UMAP0 -j umap0_Opera18days -q gpu-short


# Generate UMAP2s - Synthetic Multiplexing
# ./bash_commands/run_py.sh ./src/runables/run_synthetic_multiplexing -g -m 30000 -b 10  -a ./src/models/cytoself_vanilla/configs/config/CytoselfModelConfig ./src/datasets/configs/embeddings_data_config/EmbeddingsU2OSDatasetConfig /home/projects/hornsteinlab/Collaboration/MOmaps/outputs/figures/U2OS -j SM_U2OS

# Generate UMAP1
# ./bash_commands/run_py.sh ./src/runables/generate_umap1_vqindhist -g -m 20000 -b 10 -a ./src/models/neuroself/configs/model_config/TLNeuroselfB78NoDSModelConfig ./src/figures/figures_config/NeuronsUMAP1B78FigureConfig /home/projects/hornsteinlab/Collaboration/MOmaps/outputs/figures/manuscript/fig2/panelC/ -j umap1

# Calculate distances
# ./bash_commands/run_py.sh ./src/runables/calculate_embeddings_distances -m 40000 -a ./src/models/neuroself/configs/model_config/TLNeuroselfB78NoDSModelConfig ./src/datasets/configs/embeddings_data_config/EmbeddingsB9DatasetConfig all -q new-long -j distances

# Calculate Brenner
# ./bash_commands/run_py.sh ./src/runables/calculate_brenners -g -m 40000 -b 1 -j calc_brenner

# Test GPU
# ./bash_commands/run_py.sh ./tests/test_gpu  -m 1000 -g

# Required param
py_name=$1

# Default values
mem=15000
use_gpu=false
gmem=5
ngpu=1
queue=""
job_name="Run_$py_name"
#wait="done(343509)"
shift

while getopts "m:ga:b:n:q:j:" opt; do
  case ${opt} in
    m ) mem="$OPTARG";;
    g ) use_gpu=true;;
    b ) gmem="${OPTARG,,}";;
    n ) ngpu="$OPTARG";;
    q ) queue="$OPTARG";;
    j ) job_name="$OPTARG";;
    a ) 
        args=($OPTARG)
        while [[ ${@:$OPTIND:1} != -* && ${@:$OPTIND:1} != "" ]]; do
          args+=("${@:$OPTIND:1}")
          ((OPTIND++))
        done
        IFS=" " args="${args[*]}"
        ;;
    \? ) echo "Invalid option: -$OPTARG" 1>&2; exit 1;;
    : ) echo "Option -$OPTARG requires an argument." 1>&2; exit 1;;
  esac
done
  
if [ -z "$queue" ]; then  
  if [ "$use_gpu" = false ]; then
    queue="long"
  else
    queue="long-gpu"
  fi
fi

echo "py_name: $py_name, mem: $mem, use_gpu: $use_gpu, args: $args, gmem: $gmem, ngpu: $ngpu, queue: $queue, job_name: $job_name"

if [ "$use_gpu" = false ]
then
  bsub -n 1 -q $queue -J $job_name -B -N -u galaviram90@gmail.com -R "rusage[mem=$mem] span[hosts=1]" python $py_name.py $args
else
  bsub -n 1 -q $queue -gpu "num=${ngpu}:gmem=${gmem}G:j_exclusive=yes:aff=yes" -J $job_name -B -N -u galaviram90@gmail.com -R "rusage[mem=$mem] span[hosts=1]" python $py_name.py $args
fi


#  bsub -n 1 -w $wait -q $queue -gpu "num=${ngpu}:gmem=${gmem}G:j_exclusive=yes:aff=yes" -J $job_name -B -R "rusage[mem=$mem] span[hosts=1]" python $py_name.py $args
# bsub -n 1 -q $queue -m "public_himem_2020_hosts public_2017_hosts" -J $job_name -B -R "rusage[mem=$mem] span[hosts=1]" python $py_name.py $args
