echo "NOVA_HOME:" $NOVA_HOME

pretrained_model=/home/projects/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/pretrained_model
finetuned_model=/home/projects/hornsteinlab/Collaboration/MOmaps/outputs/vit_models/finetuned_model

embeddings_configs=./manuscript/embeddings_config

# ##################
# # pretrained_model
# ##################

# # ### OpenCell
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 30000 -b 10 \
# -a $pretrained_model $embeddings_configs/EmbeddingsOpenCellDatasetConfig 300 -q short-gpu -j emb_pretrain_opencell

# ### U2OS
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
# -a $pretrained_model $embeddings_configs/EmbeddingsU2OSDatasetConfig -q short-gpu -j emb_pretrain_U2OS

### Neurons d8
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
-a $pretrained_model $embeddings_configs/EmbeddingsB78PretrainDatasetConfig -q short-gpu -j emb_pretrain_b78

$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
-a $pretrained_model $embeddings_configs/EmbeddingsB6DatasetConfig -q short-gpu -j emb_pretrain_b6

$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
-a $pretrained_model $embeddings_configs/EmbeddingsB9DatasetConfig -q short-gpu -j emb_pretrain_b9

##################
# finetuned_model
##################

# ### OpenCell
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 30000 -b 10 \
# -a $finetuned_model $embeddings_configs/EmbeddingsOpenCellFineTuneDatasetConfig 300 -q short-gpu -j emb_opencell

### U2OS
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
# -a $finetuned_model $embeddings_configs/EmbeddingsU2OSDatasetConfig -q short-gpu -j emb_U2OS

### Neurons d8
$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
-a $finetuned_model $embeddings_configs/EmbeddingsB78DatasetConfig -q short-gpu -j emb_b78

$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
-a $finetuned_model $embeddings_configs/EmbeddingsB3DatasetConfig -q short-gpu -j emb_b3

$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
-a $finetuned_model $embeddings_configs/EmbeddingsB4DatasetConfig -q short-gpu -j emb_b4

$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
-a $finetuned_model $embeddings_configs/EmbeddingsB5DatasetConfig -q short-gpu -j emb_b5

$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
-a $finetuned_model $embeddings_configs/EmbeddingsB6DatasetConfig -q short-gpu -j emb_b6

$NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
-a $finetuned_model $embeddings_configs/EmbeddingsB9DatasetConfig -q short-gpu -j emb_b9

# ### dNLS
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
# -a $finetuned_model $embeddings_configs/EmbeddingsdNLSB3DatasetConfig -q short-gpu -j emb_dnls3

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
# -a $finetuned_model $embeddings_configs/EmbeddingsdNLSB4DatasetConfig -q short-gpu -j emb_dnls4

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
# -a $finetuned_model $embeddings_configs/EmbeddingsdNLSB5DatasetConfig -q short-gpu -j emb_dnls5

# ### Neurons d18
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
# -a $finetuned_model $embeddings_configs/EmbeddingsDay18B1DatasetConfig -q short-gpu -j emb_d18_1

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
# -a $finetuned_model $embeddings_configs/EmbeddingsDay18B2DatasetConfig -q short-gpu -j emb_d18_2

## Alyssa
# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_embeddings -g -m 20000 -b 10 \
# -a $finetuned_model $embeddings_configs/EmbeddingsAlyssaCoyneDatasetConfig -q short-gpu -j emb_alyssa
