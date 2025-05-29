from utils import registry
MODELS = registry.Registry('models')

from .paco_pipeline import PaCo
from .refinement import (
    FixedAdaptivePacoRefinementModule,  
    ResidualLoss, 
    SVDProjectionLoss
)
