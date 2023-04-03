from common.configs.base_config import BaseConfig


class PreprocessingConfig(BaseConfig):
    def __init__(self):
        super(PreprocessingConfig, self).__init__()
        
        self.input_folders = None
        self.output_folders = None
        
        self.preprocessor_class_path = None
        
        self.markers_to_include = None
        self.to_show = False
        self.nucleus_diameter = 60
        self.tile_width = 100
        self.tile_height = 100
        self.to_downsample = False
        self.to_normalize = True
        self.cellprob_threshold = 0
        self.flow_threshold = 0.4
        self.min_edge_distance = 2
        self.to_denoise = False    