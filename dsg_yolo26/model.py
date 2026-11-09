"""DSG-YOLO26 construction APIs built on Ultralytics."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from ultralytics.models.yolo.model import YOLO
from ultralytics.nn.modules import Detect
from ultralytics.nn.tasks import BaseModel, DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER, YAML
from ultralytics.utils.torch_utils import initialize_weights

from .parser import SUPPORTED_SCALES, parse_dsg_model

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "dsg-yolo26.yaml"


def _validate_build_args(scale: str, nc: int, ch: int) -> None:
    if scale not in SUPPORTED_SCALES:
        raise ValueError(f"scale must be one of {SUPPORTED_SCALES}, got '{scale}'")
    if not isinstance(nc, int) or nc <= 0:
        raise ValueError(f"nc must be a positive integer, got {nc}")
    if not isinstance(ch, int) or ch <= 0:
        raise ValueError(f"ch must be a positive integer, got {ch}")


def load_dsg_config(
    config: str | Path = DEFAULT_CONFIG, scale: str = "s", nc: int = 80
) -> dict[str, Any]:
    """Load a model YAML without relying on scale-specific duplicate files."""
    path = Path(config).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DSG-YOLO26 config not found: {path}")
    data = YAML.load(path)
    data["scale"] = scale
    data["nc"] = nc
    data["yaml_file"] = str(path)
    return data


class DSGDetectionModel(DetectionModel):
    """Ultralytics DetectionModel using the repository-local DSG parser."""

    def __init__(
        self,
        cfg: str | dict[str, Any] = DEFAULT_CONFIG,
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
    ):
        # DetectionModel's initializer calls Ultralytics' global parser. Initialize
        # the same state here with our scoped parser instead of monkeypatching it.
        BaseModel.__init__(self)
        self.yaml = load_dsg_config(cfg) if isinstance(cfg, (str, Path)) else deepcopy(cfg)
        self.yaml["channels"] = ch
        if nc is not None and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc
        self.model, self.save = parse_dsg_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.names = {i: str(i) for i in range(self.yaml["nc"])}
        self.inplace = self.yaml.get("inplace", True)

        head = self.model[-1]
        if not isinstance(head, Detect):
            raise TypeError(f"The final DSG-YOLO26 layer must be Detect, got {type(head).__name__}")
        head.inplace = self.inplace

        def forward_features(x: torch.Tensor) -> list[torch.Tensor]:
            output = self.forward(x)
            if "one2many" in output:
                output = output["one2many"]
            return output["feats"]

        # Match Ultralytics' stride discovery while avoiding an unnecessary graph.
        image_size = 256
        self.model.eval()
        head.training = True
        with torch.no_grad():
            features = forward_features(torch.zeros(1, ch, image_size, image_size))
            head.stride = torch.tensor([image_size / feature.shape[-2] for feature in features])
        self.stride = head.stride
        self.model.train()
        head.bias_init()
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")


def build_dsg_yolo26(
    scale: str = "s",
    nc: int = 80,
    ch: int = 3,
    verbose: bool = True,
    config: str | Path = DEFAULT_CONFIG,
) -> DSGDetectionModel:
    """Build an untrained DSG-YOLO26 DetectionModel.

    Args:
        scale (str): Compound model scale: n, s, m, l or x.
        nc (int): Number of detection classes.
        ch (int): Input image channels.
        verbose (bool): Print the parsed layers and model summary.
        config (str | Path): YAML configuration path.

    Returns:
        DSGDetectionModel: A randomly initialized, ready-to-inspect network.
    """
    _validate_build_args(scale, nc, ch)
    cfg = load_dsg_config(config=config, scale=scale, nc=nc)
    return DSGDetectionModel(cfg=cfg, ch=ch, nc=nc, verbose=verbose)


class DSGYOLO26(YOLO):
    """High-level Ultralytics YOLO wrapper for the custom DSG architecture.

    The PRD scope is architecture construction and forward verification. Training
    requires a matching custom trainer and is intentionally not part of this API.
    """

    def __init__(
        self,
        scale: str = "s",
        nc: int = 80,
        ch: int = 3,
        verbose: bool = False,
        config: str | Path = DEFAULT_CONFIG,
    ):
        _validate_build_args(scale, nc, ch)
        self._dsg_scale = scale
        self._dsg_nc = nc
        self._dsg_ch = ch
        super().__init__(model=str(config), task="detect", verbose=verbose)

    def _new(self, cfg: str, task: str | None = None, verbose: bool = False) -> None:
        """Build the custom DetectionModel while preserving normal YOLO metadata."""
        cfg_dict = load_dsg_config(config=cfg, scale=self._dsg_scale, nc=self._dsg_nc)
        self.cfg = cfg
        self.task = task or "detect"
        self.model = DSGDetectionModel(
            cfg=cfg_dict, ch=self._dsg_ch, nc=self._dsg_nc, verbose=verbose
        )
        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}
        self.model.task = self.task
        self.model_name = cfg
