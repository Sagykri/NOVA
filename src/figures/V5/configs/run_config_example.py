from figures.V5.configs.config import FigureV5Config

class RunConfigExample(FigureV5Config):
    def __init__(self):
        super(FigureV5Config, self).__init__()
        
        self.figures = {"1": ["c"]}