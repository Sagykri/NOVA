echo "NOVA_HOME:" $NOVA_HOME

vit_models=/home/projects/hornsteinlab/Collaboration/NOVA/outputs/vit_models

models_path=(
    $vit_models/pretrained_model
    # $vit_models/finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen
)

embeddings_configs=./manuscript/embeddings_config


for model_path in "${models_path[@]}"; do
    model_name=$(basename "$model_path")

    # OpenCell
    $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 40000 -b 10 \
    -a $model_path $embeddings_configs/EmbeddingsOpenCellDatasetConfig -q short-gpu -j emb_opencell

    # neurons d8 - NEW 
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 40000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsDay8B1DatasetConfig -q short-gpu -j emb_d8NewB1

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 40000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsDay8B2DatasetConfig -q short-gpu -j emb_d8NewB2

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 40000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsDay8B3DatasetConfig -q short-gpu -j emb_d8NewB3

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 40000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsDay8B7DatasetConfig -q short-gpu -j emb_d8NewB7

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 40000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsDay8B8DatasetConfig -q short-gpu -j emb_d8NewB8

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 40000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsDay8B9DatasetConfig -q short-gpu -j emb_d8NewB9

    ## Only FUS Lines FUS Marker
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 40000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsDay8B1DatasetConfig_FUS -q short-gpu -j emb_d8NewB1

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 40000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsDay8B2DatasetConfig_FUS -q short-gpu -j emb_d8NewB2

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 40000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsDay8B3DatasetConfig_FUS -q short-gpu -j emb_d8NewB3

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 40000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsDay8B7DatasetConfig_FUS -q short-gpu -j emb_d8NewB7

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 40000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsDay8B8DatasetConfig_FUS -q short-gpu -j emb_d8NewB8

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 40000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsDay8B9DatasetConfig_FUS -q short-gpu -j emb_d8NewB9

    #### Multiplexed  neurons d8 - NEW  ####

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_multiplexed_embeddings -g -m 20000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsDay8B1DatasetConfig_Multiplexed -q short-gpu -j emb_mul_d8New_B1

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_multiplexed_embeddings -g -m 20000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsDay8B2DatasetConfig_Multiplexed -q short-gpu -j emb_mul_d8New_B2

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_multiplexed_embeddings -g -m 20000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsDay8B3DatasetConfig_Multiplexed -q short-gpu -j emb_mul_d8New_B3
    
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_multiplexed_embeddings -g -m 20000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsDay8B7DatasetConfig_Multiplexed -q short-gpu -j emb_mul_d8New_B7

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_multiplexed_embeddings -g -m 20000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsDay8B8DatasetConfig_Multiplexed -q short-gpu -j emb_mul_d8New_B8

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_multiplexed_embeddings -g -m 20000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsDay8B8DatasetConfig_withSNCA_Multiplexed -q short-gpu -j emb_mul_d8New_withSNCA_B8

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_multiplexed_embeddings -g -m 20000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsDay8B9DatasetConfig_Multiplexed -q short-gpu -j emb_mul_d8New_B9

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_multiplexed_embeddings -g -m 20000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsDay8B9DatasetConfig_withSNCA_Multiplexed -q short-gpu -j emb_mul_d8New_withSNCA_B9

    # WO FUS MARKER
    $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_multiplexed_embeddings -g -m 20000 -b 10 \
    -a $model_path $embeddings_configs/EmbeddingsDay8B1DatasetConfig_Multiplexed_wo_FUSMarker -q short-gpu -j emb_mul_d8New_B1

    $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_multiplexed_embeddings -g -m 20000 -b 10 \
    -a $model_path $embeddings_configs/EmbeddingsDay8B2DatasetConfig_Multiplexed_wo_FUSMarker -q short-gpu -j emb_mul_d8New_B2

    $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_multiplexed_embeddings -g -m 20000 -b 10 \
    -a $model_path $embeddings_configs/EmbeddingsDay8B3DatasetConfig_Multiplexed_wo_FUSMarker -q short-gpu -j emb_mul_d8New_B3
    
    $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_multiplexed_embeddings -g -m 20000 -b 10 \
    -a $model_path $embeddings_configs/EmbeddingsDay8B7DatasetConfig_Multiplexed_wo_FUSMarker -q short-gpu -j emb_mul_d8New_B7

    $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_multiplexed_embeddings -g -m 20000 -b 10 \
    -a $model_path $embeddings_configs/EmbeddingsDay8B8DatasetConfig_Multiplexed_wo_FUSMarker -q short-gpu -j emb_mul_d8New_B8

    $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_multiplexed_embeddings -g -m 20000 -b 10 \
    -a $model_path $embeddings_configs/EmbeddingsDay8B9DatasetConfig_Multiplexed_wo_FUSMarker -q short-gpu -j emb_mul_d8New_B9

    # # ## Neurons d18
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsDay18B1DatasetConfig -q short-gpu -j emb_d18_1_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsDay18B2DatasetConfig -q short-gpu -j emb_d18_2_$model_name

    # # Alyssa
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsAlyssaCoyneDatasetConfig -q short-gpu -j emb_alyssa_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsAlyssaCoyneNEWDatasetConfig -q short-gpu -j emb_alyssa_NEW_$model_name
    
    ## Multiplexed Alyssa
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_multiplexed_embeddings -g -m 20000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsAlyssaCoyneDatasetConfig_Multiplexed -q short-gpu -j emb_alyssa_mul

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_multiplexed_embeddings -g -m 20000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsAlyssaCoyneNEWDatasetConfig_Multiplexed -q short-gpu -j emb_alyssa_NEW_mul

    # # new DNLS
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsNewdNLSB1DatasetConfig -q short-gpu -j emb_new_dnls1

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsNewdNLSB2DatasetConfig -q short-gpu -j emb_new_dnls2

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 50000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsNewdNLSB4DatasetConfig -q short-gpu -j emb_new_dnls4

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 50000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsNewdNLSB5DatasetConfig -q short-gpu -j emb_new_dnls5

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsNewdNLSB6DatasetConfig -q short-gpu -j emb_new_dnls6

    #### Multiplexed  new dNLS ####

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_multiplexed_embeddings -g -m 20000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsNewdNLSB1DatasetConfig_Multiplexed -q short-gpu -j emb_mul_new_dnls1

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_multiplexed_embeddings -g -m 20000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsNewdNLSB2DatasetConfig_Multiplexed -q short-gpu -j emb_mul_new_dnls2

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_multiplexed_embeddings -g -m 50000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsNewdNLSB4DatasetConfig_Multiplexed -q short-gpu -j emb_mul_new_dnls4

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_multiplexed_embeddings -g -m 50000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsNewdNLSB5DatasetConfig_Multiplexed -q short-gpu -j emb_mul_new_dnls5

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_multiplexed_embeddings -g -m 20000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsNewdNLSB6DatasetConfig_Multiplexed -q short-gpu -j emb_mul_new_dnls6

    # NIH
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 40000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsNIHDatasetConfig_B1 -q short-gpu -j emb_NIH1

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsNIHDatasetConfig_B2 -q short-gpu -j emb_NIH2

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsNIHDatasetConfig_B3 -q short-gpu -j emb_NIH3

     #### Multiplexed  NIH ####

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_multiplexed_embeddings -g -m 40000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsNIHDatasetConfig_WT_B1_Multiplexed -q short-gpu -j emb_mul_WT_NIH1

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_multiplexed_embeddings -g -m 20000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsNIHDatasetConfig_WT_B2_Multiplexed -q short-gpu -j emb_mul_WT_NIH2

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_multiplexed_embeddings -g -m 20000 -b 10 \
    # -a $model_path $embeddings_configs/EmbeddingsNIHDatasetConfig_WT_B3_Multiplexed -q short-gpu -j emb_mul_WT_NIH3

done

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
# -a $finetuned_model $embeddings_configs/EmbeddingsdNLSB4DatasetConfig_TDP43 -q short-gpu -j emb_dnls4_tdp

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
# -a $finetuned_model $embeddings_configs/EmbeddingsdNLSB5DatasetConfig_TDP43 -q short-gpu -j emb_dnls5_tdp



# dNLS new
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
# -a $finetuned_model $embeddings_configs/EmbeddingsNewdNLSB1DatasetConfig -q short-gpu -j emb_new_dnls1

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
# -a $finetuned_model $embeddings_configs/EmbeddingsNewdNLSB2DatasetConfig -q short-gpu -j emb_new_dnls2


# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
# -a $finetuned_model $embeddings_configs/EmbeddingsNewdNLSB4DatasetConfig -q short-gpu -j emb_new_dnls4

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
# -a $finetuned_model $embeddings_configs/EmbeddingsNewdNLSB4DatasetConfig_TDP43 -q short-gpu -j emb_new_dnls4_tdp

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 40000 -b 10 \
# -a $finetuned_model $embeddings_configs/EmbeddingsNewdNLSB5DatasetConfig_TDP43 -q short-gpu -j emb_new_dnls5_tdp

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
# -a $finetuned_model $embeddings_configs/EmbeddingsNewdNLSB4DatasetConfig_TDP43 -q short-gpu -j emb_new_dnls4_tdp


# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 40000 -b 10 \
# -a $finetuned_model $embeddings_configs/EmbeddingsNewdNLSB5DatasetConfig_TDP43 -q short-gpu -j emb_new_dnls5_tdp

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
# -a $finetuned_model $embeddings_configs/EmbeddingsNewdNLSB6DatasetConfig -q short-gpu -j emb_new_dnls6


# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 50000 -b 10 \
# -a $finetuned_model $embeddings_configs/EmbeddingsB78DatasetConfig -q short-gpu -j emb_b78



