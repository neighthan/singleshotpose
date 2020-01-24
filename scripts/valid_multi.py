from argparse import ArgumentParser
from pathlib import Path
from singleshotpose.multi_obj_pose_estimation.valid_multi import valid


def main(data_cfg_dir: str, model_cfg: str, init_weight_file: str) -> None:
    data_cfg_dir = Path(data_cfg_dir)
    for obj in ["can", "cat", "duck", "glue", "holepuncher"]:
        data_cfg = str(data_cfg_dir / f"{obj}_occlusion.data")
        valid(data_cfg, model_cfg, init_weight_file)


if __name__ == "__main__":
    proj_root = (Path(__file__).parent / "../singleshotpose").resolve()
    multi_root = proj_root / "multi_obj_pose_estimation"
    cfg_dir = multi_root / "cfg"
    default_weights = multi_root / "backup_multi/model_backup.weights"

    parser = ArgumentParser(description="SingleShotPose")
    parser.add_argument("--modelcfg", default=str(cfg_dir / "yolo-pose-multi.cfg"))
    parser.add_argument("--initweightfile", default=str(default_weights))
    parser.add_argument("--datacfgdir", default=str(cfg_dir))
    args = parser.parse_args()
    main(args.datacfgdir, args.modelcfg, args.initweightfile)
