"""Build DSG-YOLO26 and run a random-input shape smoke test.

Example (from the repository root):
    conda activate env01
    python build_model.py --scale n --nc 80 --imgsz 64
"""

from __future__ import annotations

import argparse

import torch

from dsg_yolo26 import build_dsg_yolo26
from dsg_yolo26.parser import SUPPORTED_SCALES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build untrained DSG-YOLO26 on Ultralytics")
    parser.add_argument("--scale", choices=SUPPORTED_SCALES, default="s", help="compound model scale (paper baseline: s)")
    parser.add_argument("--nc", type=int, default=80, help="number of classes")
    parser.add_argument("--ch", type=int, default=3, help="input image channels")
    parser.add_argument("--imgsz", type=int, default=64, help="square smoke-test input size (multiple of 32)")
    parser.add_argument("--skip-forward", action="store_true", help="only construct the network")
    parser.add_argument("--quiet", action="store_true", help="hide the detailed layer table")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.imgsz <= 0 or args.imgsz % 32:
        raise ValueError(f"--imgsz must be a positive multiple of 32, got {args.imgsz}")

    model = build_dsg_yolo26(scale=args.scale, nc=args.nc, ch=args.ch, verbose=not args.quiet)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    head = model.model[-1]
    print(f"Model: DSG-YOLO26{args.scale}")
    print(f"Parameters: {parameters:,} (trainable: {trainable:,})")
    print(f"Detection levels: {head.nl}; strides: {head.stride.tolist()}")
    print(
        f"YOLO26 head: one2one_branch={hasattr(head, 'one2one_cv2')}, "
        f"end2end_active={head.end2end}, reg_max={head.reg_max}, nc={head.nc}"
    )

    if not args.skip_forward:
        model.train()
        sample = torch.randn(1, args.ch, args.imgsz, args.imgsz)
        with torch.no_grad():
            output = model(sample)
        features = output["one2many"]["feats"]
        print(f"Input: {tuple(sample.shape)}")
        for level, (feature, stride) in enumerate(zip(features, head.stride), start=2):
            print(f"P{level}/{int(stride)} output: {tuple(feature.shape)}")
    print("Build and shape verification completed (random weights; no training performed).")


if __name__ == "__main__":
    main()
