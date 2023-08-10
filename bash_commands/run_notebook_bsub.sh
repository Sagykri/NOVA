notebook_name=$1
mem=$2
kernel_name=$3
use_gpu=${4,,}

if [ -z "$2" ]; then
    mem=15000
fi

if [ -z "$3" ]; then
    kernel_name='cytoself'
fi

if [ -z "$4" ]; then
    use_gpu=false
fi

echo "notebook_name: $notebook_name, mem: $mem, use_gpu: $use_gpu, kernel_name: $kernel_name"

if [ "$use_gpu" = false ]
then
  bsub -n 1 -q new-long -m "public_himem_2020_hosts public_2017_hosts" -J Run_$notebook_name -B -R "rusage[mem=$mem] span[hosts=1]" jupyter nbconvert --to notebook --inplace --execute $notebook_name.ipynb --ExecutePreprocessor.kernel_name=$kernel_name
else
  bsub -q gpu-long -gpu "num=1:gmem=5G" -J Run_$notebook_name -B -R "rusage[mem=$mem] span[hosts=1]" jupyter nbconvert --to notebook --inplace --execute $notebook_name.ipynb --ExecutePreprocessor.kernel_name=$kernel_name
fi

