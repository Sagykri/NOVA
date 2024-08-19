echo "MOMAPS_HOME:" $MOMAPS_HOME

models_folder=/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/
# models="batch78_infoNCE/ batch78_infoNCE_FMRP/ transfer_b78_freeze_all_but_attn/ transfer_b78_freeze_least_changed/ transfer_b78_freeze_mostly_changed/ transfer_b78_no_freeze/"
# models_checkpoint="checkpoints_170724_161611_633426_59193_random_rep checkpoints_280724_113218_640068_455985_with_FMRP checkpoints_250724_160110_508711_27396_freeze_all_but_attn_blocks checkpoints_250724_153840_287274_21366_freeze_least_changed_layers checkpoints_250724_153837_676628_21406_freeze_mostly_changed_layers checkpoints_240724_121100_260570_878749_no_freeze"

models="transfer_b78_freeze_least_changed/"
models_checkpoint="checkpoints_250724_153840_287274_21366_freeze_least_changed_layers"

IFS=' ' read -r -a models <<< "$models"
IFS=' ' read -r -a checkpoints <<< "$models_checkpoint"
num_classes=128
embeddings_configs=./src/embeddings/embeddings_config

# parameters order: 
# config_data output_folder num_classes model_path


# models="batch78_infoNCE_FMRP"
for index in "${!models[@]}"; do

    output_folder=$models_folder${models[$index]}
    model=$output_folder/${checkpoints[$index]}/checkpoint_best.pth
    echo $output_folder
    echo $model

#     # $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/vit_generate_embeddings -g -m 30000 -b 10 \
#     #  -a $embeddings_configs/EmbeddingsB78DatasetConfig $output_folder $num_classes $model -q gpu-short -j emb_b78

#     # $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/vit_generate_embeddings -g -m 30000 -b 10 \
#     #  -a $embeddings_configs/EmbeddingsB6DatasetConfig $output_folder $num_classes $model -q gpu-short -j emb_b6

#     # $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/vit_generate_embeddings -g -m 30000 -b 10 \
#     #  -a $embeddings_configs/EmbeddingsB9DatasetConfig $output_folder $num_classes $model -q gpu-short -j emb_b9

    $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/vit_generate_embeddings -g -m 30000 -b 10 \
     -a $embeddings_configs/EmbeddingsB3DatasetConfig $output_folder $num_classes $model -q gpu-short -j emb_b3

    $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/vit_generate_embeddings -g -m 30000 -b 10 \
     -a $embeddings_configs/EmbeddingsB4DatasetConfig $output_folder $num_classes $model -q gpu-short -j emb_b4

    $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/vit_generate_embeddings -g -m 30000 -b 10 \
     -a $embeddings_configs/EmbeddingsB5DatasetConfig $output_folder $num_classes $model -q gpu-short -j emb_b5

#     # delta NLS
#     $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/vit_generate_embeddings -g -m 60000 -b 10 \
#     -a $embeddings_configs/EmbeddingsdNLSB2DatasetConfig $output_folder $num_classes $model -q gpu-short -j emb_dNLS2

#     $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/vit_generate_embeddings -g -m 30000 -b 10 \
#      -a $embeddings_configs/EmbeddingsdNLSB3DatasetConfig $output_folder $num_classes $model -q gpu-short -j emb_dNLS3

#     $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/vit_generate_embeddings -g -m 30000 -b 10 \
#      -a $embeddings_configs/EmbeddingsdNLSB4DatasetConfig $output_folder $num_classes $model -q gpu-short -j emb_dNLS4

#     $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/vit_generate_embeddings -g -m 30000 -b 10 \
#     -a $embeddings_configs/EmbeddingsdNLSB5DatasetConfig $output_folder $num_classes $model -q gpu-short -j emb_dNLS5

#     $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/vit_generate_embeddings -g -m 60000 -b 10 \
#     -a $embeddings_configs/EmbeddingsdNLSB25DatasetConfig $output_folder $num_classes $model -q gpu-short -j emb_dNLS25_split

    # $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/vit_generate_embeddings -g -m 60000 -b 10 \
    # -a $embeddings_configs/EmbeddingsDay18B1DatasetConfig $output_folder $num_classes $model -q gpu-short -j emb_d18_1

    # $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/vit_generate_embeddings -g -m 60000 -b 10 \
    # -a $embeddings_configs/EmbeddingsDay18B2DatasetConfig $output_folder $num_classes $model -q gpu-short -j emb_d18_2

done


# ########### Opencell model ###########
# model_name=opencell_new
# model_checkpoint=checkpoints_010824_171221_851216_26417_training_pretrained_model
# num_classes=1311
# output_folder=$models_folder$model_name
# model=$output_folder/$model_checkpoint/checkpoint_best.pth
# embeddings_configs=./src/embeddings/embeddings_config

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/vit_generate_embeddings -g -m 30000 -b 10 \
# -a $embeddings_configs/EmbeddingsU2OSDatasetConfig $output_folder $num_classes $model -q gpu-short -j U2OS

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/vit_generate_embeddings -g -m 30000 -b 10 \
#     -a $embeddings_configs/EmbeddingsB78OpencellDatasetConfig $output_folder $num_classes $model -q gpu-short -j emb_b78

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/vit_generate_embeddings -g -m 30000 -b 10 \
#     -a $embeddings_configs/EmbeddingsB6DatasetConfig $output_folder $num_classes $model -q gpu-short -j emb_b6

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/vit_generate_embeddings -g -m 30000 -b 10 \
#     -a $embeddings_configs/EmbeddingsB9DatasetConfig $output_folder $num_classes $model -q gpu-short -j emb_b9

# # delta NLS
# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/vit_generate_embeddings -g -m 60000 -b 10 \
# -a $embeddings_configs/EmbeddingsdNLSB2DatasetConfig $output_folder $num_classes $model -q gpu-short -j emb_dNLS2

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/vit_generate_embeddings -g -m 30000 -b 10 \
#     -a $embeddings_configs/EmbeddingsdNLSB3DatasetConfig $output_folder $num_classes $model -q gpu-short -j emb_dNLS3

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/vit_generate_embeddings -g -m 30000 -b 10 \
#     -a $embeddings_configs/EmbeddingsdNLSB4DatasetConfig $output_folder $num_classes $model -q gpu-short -j emb_dNLS4

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/vit_generate_embeddings -g -m 30000 -b 10 \
# -a $embeddings_configs/EmbeddingsdNLSB5DatasetConfig $output_folder $num_classes $model -q gpu-short -j emb_dNLS5

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/vit_generate_embeddings -g -m 60000 -b 10 \
# -a $embeddings_configs/EmbeddingsdNLSB25DatasetConfig $output_folder $num_classes $model -q gpu-short -j emb_dNLS25_split

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/vit_generate_embeddings -g -m 60000 -b 10 \
# -a $embeddings_configs/EmbeddingsDay18B1DatasetConfig $output_folder $num_classes $model -q gpu-short -j emb_d18_1

# $MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/vit_generate_embeddings -g -m 60000 -b 10 \
# -a $embeddings_configs/EmbeddingsDay18B2DatasetConfig $output_folder $num_classes $model -q gpu-short -j emb_d18_2