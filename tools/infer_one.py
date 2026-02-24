import _init_path
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class SinglePointCloudDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, point_path, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=False,
            root_path=Path(point_path).parent,
            logger=logger,
        )
        self.point_path = Path(point_path)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        suffix = self.point_path.suffix.lower()
        if suffix == ".bin":
            points = np.fromfile(self.point_path, dtype=np.float32).reshape(-1, 4)
        elif suffix == ".npy":
            points = np.load(self.point_path)
        elif suffix == ".ply":
            try:
                import open3d as o3d
            except ImportError as exc:
                raise ImportError("Reading .ply requires open3d. Install it or use .npy/.bin.") from exc
            pcd = o3d.io.read_point_cloud(str(self.point_path))
            points = np.asarray(pcd.points, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported point cloud extension: {suffix}")

        input_dict = {"points": points, "frame_id": self.point_path.stem}
        return self.prepare_data(data_dict=input_dict)


def parse_args():
    parser = argparse.ArgumentParser(description="Single-file inference for OpenPCDet")
    parser.add_argument("--model", required=True, type=str, help="Path to checkpoint (.pth)")
    parser.add_argument("--point_cloud", required=True, type=str, help="Path to point cloud (.bin/.npy/.ply)")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="tools/cfgs/cylinder_models/pointpillar_stem.yaml",
        help="Path to model config yaml",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tools/inference_outputs/infer_one_result.json",
        help="Output JSON path for detections",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)

    logger = common_utils.create_logger()
    dataset = SinglePointCloudDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        point_path=args.point_cloud,
        logger=logger,
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.model, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    with torch.no_grad():
        data_dict = dataset[0]
        batch_dict = dataset.collate_batch([data_dict])
        load_data_to_gpu(batch_dict)
        pred_dicts, _ = model.forward(batch_dict)

    pred = pred_dicts[0]
    labels = pred["pred_labels"].detach().cpu().numpy().tolist()
    scores = pred["pred_scores"].detach().cpu().numpy().tolist()
    boxes = pred["pred_boxes"].detach().cpu().numpy().tolist()

    class_names = cfg.CLASS_NAMES
    class_names_per_box = [class_names[idx - 1] if 0 < idx <= len(class_names) else str(idx) for idx in labels]

    result = {
        "checkpoint": str(Path(args.model).resolve()),
        "point_cloud": str(Path(args.point_cloud).resolve()),
        "num_detections": len(scores),
        "detections": [
            {
                "label_id": labels[i],
                "label_name": class_names_per_box[i],
                "score": scores[i],
                "box_3d": boxes[i],
            }
            for i in range(len(scores))
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    logger.info("Inference complete")
    logger.info("Detections: %d", len(scores))
    logger.info("Saved: %s", output_path.resolve())


if __name__ == "__main__":
    main()
