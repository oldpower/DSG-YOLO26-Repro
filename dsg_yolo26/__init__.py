"""Local, untrained DSG-YOLO26 reconstruction based on Ultralytics."""

from .model import DEFAULT_CONFIG, DSGDetectionModel, DSGYOLO26, build_dsg_yolo26
from .modules import DPSE, SGCF, DetailPreservingShallowEnhancement, ScaleGatedCrossLevelFusion
from .ssr import ScaleSensitiveRegression

__all__ = (
    "DEFAULT_CONFIG",
    "DPSE",
    "DSGDetectionModel",
    "DSGYOLO26",
    "SGCF",
    "DetailPreservingShallowEnhancement",
    "ScaleGatedCrossLevelFusion",
    "ScaleSensitiveRegression",
    "build_dsg_yolo26",
)

__version__ = "0.1.0"
