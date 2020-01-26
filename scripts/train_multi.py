import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
import torch
import torch.optim as optim
from singleshotpose.multi_obj_pose_estimation.train_multi import train_loop


if __name__ == "__main__":
    proj_root = (Path(__file__).parent / "../singleshotpose").resolve()
    multi_root = proj_root / "multi_obj_pose_estimation"
    cfg_dir = multi_root / "cfg"
    default_weights = multi_root / "backup_multi/init.weights"

    parser = ArgumentParser(description="SingleShotPose")
    parser.add_argument("--datacfg", default=str(cfg_dir / "occlusion.data"))
    parser.add_argument("--modelcfg", default=str(cfg_dir / "yolo-pose-multi.cfg"))
    parser.add_argument("--initweightfile", default=str(default_weights))
    parser.add_argument("--pretrain_num_epochs", type=int, default=0)
    args = parser.parse_args()
    train_loop(
        args.datacfg, args.modelcfg, args.initweightfile, args.pretrain_num_epochs
    )
