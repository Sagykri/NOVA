echo "MOMAPS_HOME:" $MOMAPS_HOME

cytoself=./src/models/cytoself_vanilla/configs/config/CytoselfModelConfig
neuroself=./src/models/neuroself/configs/model_config/TLNeuroselfB78NoDSModelConfig
deltaNLS=./src/models/neuroself/configs/model_config/TLNeuroselfdeltaNLSB25ModelConfig
folder_path=/home/labs/hornsteinlab/Collaboration/MOmaps/outputs/figures/tmp_sagy/110124
figures_configs=./src/figures/figures_config

#generate_umaps
#run_synthetic_multiplexing
#generate_umap1_vqindhist

# Fig1A
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps -g -m 30000 -b 10 \
 -a $cytoself $figures_configs/U2OSUMAP0StressFigureConfig  $folder_path/fig1/A -q gpu-short -j fig1A

# Fig1C
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps -g -m 30000 -b 10 \
 -a $cytoself $figures_configs/NeuronsUMAP0B6StressFigureConfig  $folder_path/fig1/C/cyto -q gpu-short -j fig1C_cyto

# Fig1C
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps -g -m 30000 -b 10 \
 -a $neuroself $figures_configs/NeuronsUMAP0B6StressFigureConfig  $folder_path/fig1/C/neuroself -q gpu-short -j fig1C_neuroself

# Fig2B
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umap1_vqindhist -g -m 30000 -b 10 \
 -a $neuroself $figures_configs/NeuronsUMAP1B78FigureConfig  $folder_path/fig2/B -q gpu-short -j fig2B

# Fig3A
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps -g -m 30000 -b 10 \
 -a $neuroself $figures_configs/Fig3AConfig  $folder_path/fig3/A -q gpu-short -j fig3A

# Fig4C
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps -g -m 30000 -b 10 \
 -a $deltaNLS $figures_configs/Fig4CConfig  $folder_path/fig4/C -q gpu-short -j fig4c

# Fig5C
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/run_synthetic_multiplexing -g -m 30000 -b 10 \
 -a $neuroself $figures_configs/Fig5CB6Config  $folder_path/fig5/C -q gpu-short -j fig5c

# Fig5D
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps -g -m 30000 -b 10 \
 -a $neuroself $figures_configs/Fig5DB6Config  $folder_path/fig5/D/b6 -q gpu-short -j fig5d_b6

# Fig5D
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps -g -m 30000 -b 10 \
 -a $neuroself $figures_configs/Fig5DB9Config  $folder_path/fig5/D/b9 -q gpu-short -j fig5d_b9

# Fig5E
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/run_synthetic_multiplexing -g -m 30000 -b 10 \
 -a $neuroself $figures_configs/Fig5EB6Config  $folder_path/fig5/E -q gpu-short -j fig5e

# Fig5F
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/run_synthetic_multiplexing -g -m 30000 -b 10 \
 -a $neuroself $figures_configs/Fig5FB6Config  $folder_path/fig5/F/b6 -q gpu-short -j fig5f_b6

# Fig5F - SCNA
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/run_synthetic_multiplexing -g -m 30000 -b 10 \
 -a $neuroself $figures_configs/Fig5FB6SCNAConfig  $folder_path/fig5/F/scna/b6 -q gpu-short -j fig5f_scna_b6

# Fig5F - SCNA
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/run_synthetic_multiplexing -g -m 30000 -b 10 \
 -a $neuroself $figures_configs/Fig5FB9SCNAConfig  $folder_path/fig5/F/scna/b9 -q gpu-short -j fig5f_scna_b9

# Sup Fig2 A
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps -g -m 30000 -b 10 \
 -a $cytoself $figures_configs/FigSup2AConfig  $folder_path/sup_fig2/A -q gpu-short -j sup_fig2A

# Sup Fig4 A
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps -g -m 30000 -b 10 \
 -a $neuroself $figures_configs/FigSup4AConfig  $folder_path/sup_fig4/A -q gpu-short -j sup_fig4A

# Sup Fig5 A
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/generate_umaps -g -m 30000 -b 10 \
 -a $deltaNLS $figures_configs/FigSup5AConfig  $folder_path/sup_fig5/A -q gpu-short -j sup_fig5A

# Sup Fig6 A
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/run_synthetic_multiplexing -g -m 30000 -b 10 \
 -a $neuroself $figures_configs/FigSup6AConfig  $folder_path/sup_fig6/A -q gpu-short -j sup_fig6A

# Sup Fig6 B
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/run_synthetic_multiplexing -g -m 30000 -b 10 \
 -a $neuroself $figures_configs/FigSup6BB6Config  $folder_path/sup_fig6/B/b6 -q gpu-short -j sup_fig6B_b6

# Sup Fig6 B
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/run_synthetic_multiplexing -g -m 30000 -b 10 \
 -a $neuroself $figures_configs/FigSup6BB9Config  $folder_path/sup_fig6/B/b9 -q gpu-short -j sup_fig6B_b9

# Sup Fig6 C
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/run_synthetic_multiplexing -g -m 30000 -b 10 \
 -a $neuroself $figures_configs/FigSup6CConfig  $folder_path/sup_fig6/C -q gpu-short -j sup_fig6C

# # Sup Fig6 D
$MOMAPS_HOME/bash_commands/run_py.sh $MOMAPS_HOME/src/runables/run_synthetic_multiplexing -g -m 30000 -b 10 \
 -a $deltaNLS $figures_configs/FigSup6DConfig  $folder_path/sup_fig6/D -q gpu-short -j sup_fig6D