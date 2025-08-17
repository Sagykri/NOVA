echo "NOVA_HOME:" $NOVA_HOME

vit_models=/home/projects/hornsteinlab/Collaboration/NOVA/outputs/vit_models

models_path=(
    # $vit_models/pretrained_model
    $vit_models/finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen
)


data_configs=./manuscript/manuscript_figures_data_config
plot_configs=./manuscript/manuscript_plot_config


for model_path in "${models_path[@]}"; do
    model_name=$(basename "$model_path")

    # # new dNLS

    $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 30000 \
    -a $model_path $data_configs/newDNLSUMAP0DatasetConfig_Hits $plot_configs/UMAP0dNLSPlotConfig -q short -j umap0_new_dnls_Hits

    $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 30000 \
    -a $model_path $data_configs/newDNLSUMAP0DatasetConfig_Hits_With_WT $plot_configs/UMAP0dNLSPlotConfig -q short -j umap0_new_dnls_Hits_with_WT
    
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/newDNLSUMAP0B1DatasetConfig $plot_configs/UMAP0dNLSPlotConfig -q short -j umap0_new_dnls_b1

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/newDNLSUMAP0B2DatasetConfig $plot_configs/UMAP0dNLSPlotConfig -q short -j umap0_new_dnls_b2

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/newDNLSUMAP0B3DatasetConfig $plot_configs/UMAP0dNLSPlotConfig -q short -j umap0_new_dnls_b3

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 30000 \
    # -a $model_path $data_configs/newDNLSUMAP0B4DatasetConfig $plot_configs/UMAP0dNLSPlotConfig -q short -j umap0_new_dnls_b4

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newDNLSUMAP0B5DatasetConfig $plot_configs/UMAP0dNLSPlotConfig -q short -j umap0_new_dnls_b5

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/newDNLSUMAP0B6DatasetConfig $plot_configs/UMAP0dNLSPlotConfig -q short -j umap0_new_dnls_b6

    # # Umap1 WT Untreated
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newDNLSUntreatedUMAP1DatasetConfig $plot_configs/UMAP1PlotConfig -q short -j umap1_new_dnls

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/newDNLSUntreatedUMAP1DatasetConfig_B1 $plot_configs/UMAP1PlotConfig -q short -j umap1_new_dnls_b1

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/newDNLSUntreatedUMAP1DatasetConfig_B2 $plot_configs/UMAP1PlotConfig -q short -j umap1_new_dnls_b2

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/newDNLSUntreatedUMAP1DatasetConfig_B3 $plot_configs/UMAP1PlotConfig -q short -j umap1_new_dnls_b3

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/newDNLSUntreatedUMAP1DatasetConfig_B4 $plot_configs/UMAP1PlotConfig -q short -j umap1_new_dnls_b4

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/newDNLSUntreatedUMAP1DatasetConfig_B5 $plot_configs/UMAP1PlotConfig -q short -j umap1_new_dnls_b5

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/newDNLSUntreatedUMAP1DatasetConfig_B6 $plot_configs/UMAP1PlotConfig -q short -j umap1_new_dnls_b6

    # # UMAP2
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newDNLSFigureConfig_UMAP2 $plot_configs/UMAP2dNLSPlotConfig -q short -j umap2_new_dnls

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newDNLSFigureConfig_UMAP2_B1 $plot_configs/UMAP2dNLSPlotConfig -q short -j umap2_new_dnls_b1

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newDNLSFigureConfig_UMAP2_B2 $plot_configs/UMAP2dNLSPlotConfig -q short -j umap2_new_dnls_b2

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newDNLSFigureConfig_UMAP2_B3 $plot_configs/UMAP2dNLSPlotConfig -q short -j umap2_new_dnls_b3

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newDNLSFigureConfig_UMAP2_B4 $plot_configs/UMAP2dNLSPlotConfig -q short -j umap2_new_dnls_b4

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newDNLSFigureConfig_UMAP2_B5 $plot_configs/UMAP2dNLSPlotConfig -q short -j umap2_new_dnls_b5

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newDNLSFigureConfig_UMAP2_B6 $plot_configs/UMAP2dNLSPlotConfig -q short -j umap2_new_dnls_b6

    # ### Effect size
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_effect_sizes -m 1000 -a \
    # $model_path $data_configs/dNLSNewEffectsFigureConfig  $plot_configs/DistancesdNLSPlotConfig -q short -j dnls_new

    

    ###############
    # new INDI
    ###############

    # # UMAP1
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 100000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP1 $plot_configs/UMAP1PlotConfig -q short -j u1_nD8

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 80000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP1_NIHMarkers $plot_configs/UMAP1PlotConfig -q short -j u1_nD8

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP1_B1 $plot_configs/UMAP1PlotConfig -q short -j u1_nD8_B1

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP1_B2 $plot_configs/UMAP1PlotConfig -q short -j u1_nD8_B2

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP1_B3 $plot_configs/UMAP1PlotConfig -q short -j u1_nD8_B3

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP1_B7 $plot_configs/UMAP1PlotConfig -q short -j u1_nD8_B7

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP1_B8 $plot_configs/UMAP1PlotConfig -q short -j u1_nD8_B8

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP1_B9 $plot_configs/UMAP1PlotConfig -q short -j u1_nD8_B9
    
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP1_B10 $plot_configs/UMAP1PlotConfig -q short -j u1_nD8_B10

    # # UMAP0 - one line one marker - UNUSED
    # # B2
    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B2_WT $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B2_WT$model_name

    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B2_TDP43 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B2_TDP43$model_name

    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B2_TBK1 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B2_TBK1$model_name

    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B2_OPTN $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B2_OPTN$model_name

    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B2_FUSRevertant $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B2_FUSRev$model_name

    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B2_FUSHomozygous $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B2_FUSHom$model_name

    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B2_FUSHeterozygous $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B2_FUSHet$model_name


    # # UMAP0 - WT FUSREV  - UNUSED
    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B1_WT_FUSREV $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B1_WT_FUSREV$model_name

    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B3_WT_FUSREV $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B3_WT_FUSREV$model_name

    # # UMAP0 - one marker two lines (WT and mutated line)

    # B1 - B10:
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 70000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_allBatches_WT_TDP43 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_WT_TDP43
    
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 70000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_allBatches_WT_OPTN $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_WT_OPTN
    
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 70000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_allBatches_WT_TBK1 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_WT_TBK1
    
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 70000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_allBatches_WT_FUSRev $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_WT_FUSRev
    
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 70000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_allBatches_WT_FUSHet $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_WT_FUSHet
    
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 70000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_allBatches_WT_FUSHom $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_WT_FUSHom
    

    # B1:
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B1_WT_TDP43 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B1_WT_TDP43$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B1_WT_OPTN $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B1_WT_OPTN$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B1_WT_TBK1 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B1_WT_TBK1$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B1_WT_FUSRev $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B1_WT_FUSRev$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B1_WT_FUSHet $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B1_WT_FUSHet$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B1_WT_FUSHom $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B1_WT_FUSHom$model_name

    # # B2:
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B2_WT_TDP43 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B2_WT_TDP43$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B2_WT_OPTN $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B2_WT_OPTN$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B2_WT_TBK1 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B2_WT_TBK1$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B2_WT_FUSRev $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B2_WT_FUSRev$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B2_WT_FUSHet $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B2_WT_FUSHet$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B2_WT_FUSHom $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B2_WT_FUSHom$model_name

    # # B3:
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B3_WT_TDP43 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B3_WT_TDP43$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B3_WT_OPTN $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B3_WT_OPTN$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B3_WT_TBK1 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B3_WT_TBK1$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B3_WT_FUSRev $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B3_WT_FUSRev$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B3_WT_FUSHet $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B3_WT_FUSHet$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B3_WT_FUSHom $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B3_WT_FUSHom$model_name

    # # B7:
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
	# -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B7_WT_TDP43 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B7_WT_TDP43$model_name

	# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
	# -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B7_WT_OPTN $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B7_WT_OPTN$model_name

	# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
	# -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B7_WT_TBK1 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B7_WT_TBK1$model_name

	# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
	# -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B7_WT_FUSRev $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B7_WT_FUSRev$model_name

	# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
	# -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B7_WT_FUSHet $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B7_WT_FUSHet$model_name

	# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
	# -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B7_WT_FUSHom $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B7_WT_FUSHom$model_name

    # # B8:
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
	# -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B8_WT_TDP43 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B8_WT_TDP43$model_name

	# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
	# -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B8_WT_OPTN $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B8_WT_OPTN$model_name

	# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
	# -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B8_WT_TBK1 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B8_WT_TBK1$model_name

	# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
	# -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B8_WT_FUSRev $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B8_WT_FUSRev$model_name

	# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
	# -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B8_WT_FUSHet $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B8_WT_FUSHet$model_name

	# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
	# -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B8_WT_FUSHom $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B8_WT_FUSHom$model_name


    # # B9:
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
	# -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B9_WT_TDP43 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B9_WT_TDP43$model_name

	# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
	# -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B9_WT_OPTN $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B9_WT_OPTN$model_name

	# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
	# -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B9_WT_TBK1 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B9_WT_TBK1$model_name

	# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
	# -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B9_WT_FUSRev $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B9_WT_FUSRev$model_name

	# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
	# -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B9_WT_FUSHet $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B9_WT_FUSHet$model_name

	# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
	# -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B9_WT_FUSHom $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B9_WT_FUSHom$model_name

    # # B10:
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
	# -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B10_WT_TDP43 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B10_WT_TDP43$model_name

	# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
	# -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B10_WT_OPTN $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B10_WT_OPTN$model_name

	# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
	# -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B10_WT_TBK1 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B10_WT_TBK1$model_name

	# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
	# -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B10_WT_FUSRev $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B10_WT_FUSRev$model_name

	# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
	# -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B10_WT_FUSHet $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B10_WT_FUSHet$model_name

	# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
	# -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B10_WT_FUSHom $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_B10_WT_FUSHom$model_name

    ######
    ## UMAP0 all lines from all batches

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 100000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_ALS_UMAP0_allBatches_allALSLines $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_allBatches_ALS



    # ######

    # # # UMAP0 - stress - UNUSED
    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0 $plot_configs/UMAP0StressPlotConfig -q short -j u0_nD8_stress_$model_name

    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B1 $plot_configs/UMAP0StressPlotConfig -q short -j u0_nD8_stress_B1_$model_name

    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B2 $plot_configs/UMAP0StressPlotConfig -q short -j u0_nD8_stress_B2_$model_name

    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B3 $plot_configs/UMAP0StressPlotConfig -q short -j u0_nD8_stress_B3_$model_name

    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B7 $plot_configs/UMAP0StressPlotConfig -q short -j u0_nD8_stress_B7_$model_name

    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B8 $plot_configs/UMAP0StressPlotConfig -q short -j u0_nD8_stress_B8_$model_name

    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B9 $plot_configs/UMAP0StressPlotConfig -q short -j u0_nD8_stress_B9_$model_name

    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP0_B10 $plot_configs/UMAP0StressPlotConfig -q short -j u0_nD8_stress_B10_$model_name


    # # # UMAP0 - ALS - UNUSED
    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 70000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_ALS_UMAP0 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_ALS_$model_name

    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_ALS_UMAP0_B1 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_ALS_B1_$model_name

    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_ALS_UMAP0_B2 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_ALS_B2_$model_name

    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_ALS_UMAP0_B3 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_ALS_B3_$model_name

    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_ALS_UMAP0_B7 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_ALS_B7_$model_name

    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_ALS_UMAP0_B8 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_ALS_B8_$model_name

    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_ALS_UMAP0_B9 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_ALS_B9_$model_name

    # # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # # -a $model_path $data_configs/newNeuronsD8FigureConfig_ALS_UMAP0_B10 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_ALS_B10_$model_name

    # # # UMAP0 - FUS lines
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 40000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_FUSLines_UMAP0 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_FUSLines_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_FUSLines_UMAP0_B1 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_FUSLines_B1_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_FUSLines_UMAP0_B2 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_FUSLines_B2_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_FUSLines_UMAP0_B3 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_FUSLines_B3_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_FUSLines_UMAP0_B7 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_FUSLines_B7_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_FUSLines_UMAP0_B8 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_FUSLines_B8_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_FUSLines_UMAP0_B9 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_FUSLines_B9_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_FUSLines_UMAP0_B10 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_nD8_FUSLines_B10_$model_name


    # # UMAP2
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 150000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_B1 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B1_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_B1_wo_FUSMarker $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B1_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_B2 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B2_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_B2_wo_FUSMarker $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B2_$model_name


    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_B3 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B3_$model_name
    
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_B3_wo_FUSMarker $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B3_$model_name


    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_B7 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B7_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_B7_wo_FUSMarker $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B7_$model_name


    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_B8 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B8_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_B8_wo_FUSMarker $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B8_$model_name


    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_B9 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B9_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_B9_wo_FUSMarker $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B9_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_B10 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B10_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_B10_wo_FUSMarker $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B10_$model_name

    # UMAP2 hits marker
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_Hits_B1 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B1_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_Hits_B2 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B2_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_Hits_B3 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B3_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_Hits_B7 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B7_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_Hits_B8 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B8_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_Hits_B9 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B9_$model_name

    ## UMAP2 without hits markers
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_wo_Hits_B1 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B1_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_wo_Hits_B2 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B2_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_wo_Hits_B3 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B3_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_wo_Hits_B7 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B7_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_wo_Hits_B8 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B8_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_wo_Hits_B9 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B9_$model_name


    # UMAP2 FUS lines
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_FUSLines_B1_wo_FUSMarker $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B1

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_FUSLines_B2_wo_FUSMarker $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B2

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_FUSLines_B3_wo_FUSMarker $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B3

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_FUSLines_B7_wo_FUSMarker $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B7

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_FUSLines_B8_wo_FUSMarker $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B8

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_FUSLines_B9_wo_FUSMarker $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B9

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_FUSLines_B10_wo_FUSMarker $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_B10


    # UMAP2 - 6 markers (cell painting)
    # DAPI, Calreticulin, NCL, mitotracker, Phalloidin, GM130
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 80000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_CellPainting $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_cpaint

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 25000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_CellPainting_B1 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_cpaint_b1

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 25000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_CellPainting_B2 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_cpaint_b2

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 25000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_CellPainting_B3 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_cpaint_b3

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 25000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_CellPainting_B7 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_cpaint_b7

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 25000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_CellPainting_B8 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_cpaint_b8

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 25000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_CellPainting_B9 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_cpaint_b9

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 25000 \
    # -a $model_path $data_configs/newNeuronsD8FigureConfig_UMAP2_CellPainting_B10 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_nD8_cpaint_b10


    ## Effect size
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_effect_sizes -m 1000 -a \
    # $model_path $data_configs/NeuronsEffectsFigureConfig  $plot_configs/DistancesNeuronsALSPlotConfig -q short -j effect_sizes


    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_effect_sizes -m 1000 -a \
    # $model_path $data_configs/NeuronsEffectsFigureConfig_with_DAPI  $plot_configs/DistancesNeuronsALSPlotConfig -q short -j effect_sizes
    

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_effect_sizes -m 1000 -a \
    # $model_path $data_configs/NeuronsEffectsFigureConfig_WO10  $plot_configs/DistancesNeuronsALSPlotConfig -q short -j effect_sizes


    ###
    ## Alyssa old
    ##

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/AlyssaCoyneUMAP2FigureConfig $plot_configs/UMAP2AlyssaCoyneColorByGroupPlotConfig -q short -j umap2_alyssa_groups

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/AlyssaCoyneUMAP2FigureConfig $plot_configs/UMAP2AlyssaCoyneColorByPatientColorControlsPlotConfig -q short -j umap2_alyssa_control

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/AlyssaCoyneUMAP2FigureConfig $plot_configs/UMAP2AlyssaCoyneColorByPatientColorsALSPositivePlotConfig -q short -j umap2_alyssa_pos

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/AlyssaCoyneUMAP2FigureConfig $plot_configs/UMAP2AlyssaCoyneColorByPatientColorsALSNegativePlotConfig -q short -j umap2_alyssa_neg

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/AlyssaCoyneUMAP2FigureConfig $plot_configs/UMAP2AlyssaCoyneColorByPatientColorC9PlotConfig -q short -j umap2_alyssa_c9



    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/AlyssaCoyneUMAP0FigureConfig $plot_configs/UMAP0AlyssaCoyneColorByGroupPlotConfig -q short -j umap0_alyssa_groups

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/AlyssaCoyneUMAP0FigureConfig $plot_configs/UMAP0AlyssaCoyneColorByPatientPlotConfig -q short -j umap0_alyssa_patient



    ###
    ### Alyssa new
    ##
    # umap1
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newAlyssaFigureConfig_UMAP1_B1_C9_CS2YNL $plot_configs/UMAP1PlotConfig -q short -j u1_nAlys_CS2YNL_$model_name
    
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newAlyssaFigureConfig_UMAP1_B1_C9_CS7VCZ $plot_configs/UMAP1PlotConfig -q short -j u1_nAlys_CS7VCZ_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newAlyssaFigureConfig_UMAP1_B1_C9_CS8RFT $plot_configs/UMAP1PlotConfig -q short -j u1_nAlys_CS8RFT_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newAlyssaFigureConfig_UMAP1_B1_Ctrl_EDi022 $plot_configs/UMAP1PlotConfig -q short -j u1_nAlys_EDi022_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newAlyssaFigureConfig_UMAP1_B1_Ctrl_EDi029 $plot_configs/UMAP1PlotConfig -q short -j u1_nAlys_EDi029_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newAlyssaFigureConfig_UMAP1_B1_Ctrl_EDi037 $plot_configs/UMAP1PlotConfig -q short -j u1_nAlys_EDi037_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newAlyssaFigureConfig_UMAP1_B1_SALSNegative_CS0ANK $plot_configs/UMAP1PlotConfig -q short -j u1_nAlys_CS0ANK_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newAlyssaFigureConfig_UMAP1_B1_SALSNegative_CS0JPP $plot_configs/UMAP1PlotConfig -q short -j u1_nAlys_CS0JPP_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newAlyssaFigureConfig_UMAP1_B1_SALSNegative_CS6ZU8 $plot_configs/UMAP1PlotConfig -q short -j u1_nAlys_CS6ZU8_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newAlyssaFigureConfig_UMAP1_B1_SALSPositive_CS2FN3 $plot_configs/UMAP1PlotConfig -q short -j u1_nAlys_CS2FN3_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newAlyssaFigureConfig_UMAP1_B1_SALSPositive_CS4ZCD $plot_configs/UMAP1PlotConfig -q short -j u1_nAlys_CS4ZCD_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/newAlyssaFigureConfig_UMAP1_B1_SALSPositive_CS7TN6 $plot_configs/UMAP1PlotConfig -q short -j u1_nAlys_CS7TN6_$model_name

    # # # # umap1 patient combined
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 40000 \
    # -a $model_path $data_configs/newAlyssaFigureConfig_UMAP1_B1_C9 $plot_configs/UMAP1PlotConfig -q short -j u1_nAlys_C9_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 40000 \
    # -a $model_path $data_configs/newAlyssaFigureConfig_UMAP1_B1_Ctrl $plot_configs/UMAP1PlotConfig -q short -j u1_nAlys_Ctrl_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 40000 \
    # -a $model_path $data_configs/newAlyssaFigureConfig_UMAP1_B1_SALSNegative $plot_configs/UMAP1PlotConfig -q short -j u1_nAlys_SALSNegative_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 40000 \
    # -a $model_path $data_configs/newAlyssaFigureConfig_UMAP1_B1_SALSPositive $plot_configs/UMAP1PlotConfig -q short -j u1_nAlys_SALSPositive_$model_name

    # # # umap2
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newAlyssaFigureConfig_UMAP2_B1 $plot_configs/UMAP2AlyssaCoyneColorByGroupPlotConfig -q short -j u2_nAlys_g_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newAlyssaFigureConfig_UMAP2_B1_with_patientID $plot_configs/UMAP2AlyssaCoyneColorByPatientPlotConfig -q short -j u2_nAlys_p_$model_name

    # Per plate
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newAlyssaFigureConfig_UMAP2_B1_P1 $plot_configs/UMAP2AlyssaCoyneColorByGroupPlotConfig -q short -j u2_nAlys_p1_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newAlyssaFigureConfig_UMAP2_B1_P2 $plot_configs/UMAP2AlyssaCoyneColorByGroupPlotConfig -q short -j u2_nAlys_p2_$model_name

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newAlyssaFigureConfig_UMAP2_B1_P3 $plot_configs/UMAP2AlyssaCoyneColorByGroupPlotConfig -q short -j u2_nAlys_p3_$model_name

    ## UMAP2 Only 4 markers (DCP1A, TDP43, MAP2, DAPI) 
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 50000 \
    # -a $model_path $data_configs/newAlyssaFigureConfig_UMAP2_4Markers_with_patientID $plot_configs/UMAP2AlyssaCoyneColorByPatientPlotConfig -q short -j u2_nAlys_p_4markers


    # # UMAP0
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/newAlyssaFigureConfig_UMAP0_B1 $plot_configs/UMAP0newAlyssaCoyne -q short -j umap0_newAlyssa_patient

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 10000 \
    # -a $model_path $data_configs/newAlyssaFigureConfig_UMAP0_B1_per_gene_group $plot_configs/UMAP0newAlyssaCoyne -q short -j umap0_newAlyssa_group


    #### Effect size ###

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_effect_sizes_alyssa -m 1000 \
    # -a $model_path $data_configs/AlyssaCoyneEffectsFigureConfig $plot_configs/DistancesAlyssaCoynePlotConfig -q short -j alyssa_dist

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_effect_sizes_alyssa_new -m 1000 \
    # -a $model_path $data_configs/AlyssaCoyneNEWEffectsFigureConfig $plot_configs/DistancesAlyssaCoynePlotConfig -q short -j new_alyssa_dist


    #############


    ###
    ## NIH ###
    ####

    # UMAP1
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 30000 \
    # -a $model_path $data_configs/NIH_UMAP1_DatasetConfig $plot_configs/UMAP1PlotConfig -q short -j u1_NIH1

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/NIH_UMAP1_DatasetConfig_B1 $plot_configs/UMAP1PlotConfig -q short -j u1_NIH1_B1

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/NIH_UMAP1_DatasetConfig_B2 $plot_configs/UMAP1PlotConfig -q short -j u1_NIH1_B2

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/NIH_UMAP1_DatasetConfig_B3 $plot_configs/UMAP1PlotConfig -q short -j u1_NIH1_B3

    # # UMAP0 stress
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/NIH_UMAP0_Stress_DatasetConfig $plot_configs/UMAP0StressPlotConfig -q short -j u0_NIH1_stress

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/NIH_UMAP0_Stress_DatasetConfig_B1 $plot_configs/UMAP0StressPlotConfig -q short -j u0_NIH1_stress_B1

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/NIH_UMAP0_Stress_DatasetConfig_B2 $plot_configs/UMAP0StressPlotConfig -q short -j u0_NIH1_stress_B2

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/NIH_UMAP0_Stress_DatasetConfig_B3 $plot_configs/UMAP0StressPlotConfig -q short -j u0_NIH1_stress_B3

    # # umap0 FUS
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 30000 \
    # -a $model_path $data_configs/NIH_UMAP0_FUS_DatasetConfig $plot_configs/UMAP0ALSPlotConfig -q short -j u0_NIH1_FUS

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/NIH_UMAP0_FUS_DatasetConfig_B1 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_NIH1_FUS_B1

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/NIH_UMAP0_FUS_DatasetConfig_B2 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_NIH1_FUS_B2

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/NIH_UMAP0_FUS_DatasetConfig_B3 $plot_configs/UMAP0ALSPlotConfig -q short -j u0_NIH1_FUS_B3

    # # # Umap2 FUS
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 30000 \
    # -a $model_path $data_configs/NIH_UMAP2_FUS_DatasetConfig $plot_configs/UMAP2ALSPlotConfig -q short -j u2_NIH1_FUS

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/NIH_UMAP2_FUS_DatasetConfig_B1 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_NIH1_FUS_B1

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/NIH_UMAP2_FUS_DatasetConfig_B2 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_NIH1_FUS_B2

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/NIH_UMAP2_FUS_DatasetConfig_B3 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_NIH1_FUS_B3

    # # UMAP2 FUS without FUS marker
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 30000 \
    # -a $model_path $data_configs/NIH_UMAP2_FUS_DatasetConfig_wo_FUS_Marker $plot_configs/UMAP2ALSPlotConfig -q short -j u2_NIH1_FUS

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/NIH_UMAP2_FUS_DatasetConfig_wo_FUS_Marker_B1 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_NIH1_FUS_B1

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/NIH_UMAP2_FUS_DatasetConfig_wo_FUS_Marker_B2 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_NIH1_FUS_B2

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/NIH_UMAP2_FUS_DatasetConfig_wo_FUS_Marker_B3 $plot_configs/UMAP2ALSPlotConfig -q short -j u2_NIH1_FUS_B3

    # # UMAP2 stress
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 30000 \
    # -a $model_path $data_configs/NIH_UMAP2_Stress_DatasetConfig $plot_configs/UMAP2StressPlotConfig -q short -j u2_NIH1_stress

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/NIH_UMAP2_Stress_DatasetConfig_B1 $plot_configs/UMAP2StressPlotConfig -q short -j u2_NIH1_stress_B1

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/NIH_UMAP2_Stress_DatasetConfig_B2 $plot_configs/UMAP2StressPlotConfig -q short -j u2_NIH1_stress_B2

    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/generate_umaps_and_plot -m 20000 \
    # -a $model_path $data_configs/NIH_UMAP2_Stress_DatasetConfig_B3 $plot_configs/UMAP2StressPlotConfig -q short -j u2_NIH1_stress_B3

    ## Effect size
    # $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_effect_sizes -m 1000 -a \
    # $model_path $data_configs/NIHEffectsFigureConfig  $plot_configs/DistancesNeuronsALSPlotConfig -q short -j effect_sizes


    ###

done
# # #######################
# # # new neurons day8 (Opera)
# # # #######################

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_effect_sizes -m 1000 -a \
# $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $data_configs/NeuronsEffectsFigureConfig  $plot_configs/DistancesNeuronsALSPlotConfig -q short -j effect_sizes

# $NOVA_HOME/runnables/run.sh $NOVA_HOME/runnables/plot_effect_sizes_multiplex -m 3000 -a \
# $finetunedModel_MLPHead_acrossBatches_B56789_80pct_frozen $data_configs/NeuronsMultiplexedEffectsFigureConfig  $plot_configs/DistancesNeuronsALSPlotConfig -q short -j effect_multi_plot


