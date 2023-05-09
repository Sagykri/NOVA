from figures.V5.configs.config import FigureV5Config

class RunConfigExample(FigureV5Config):
    def __init__(self):
        super().__init__()
        
        self.FIGURES = {"1": ["c"]}