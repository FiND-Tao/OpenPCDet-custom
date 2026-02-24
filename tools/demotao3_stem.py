#%%
import argparse
import glob
from pathlib import Path
import os


try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        a=str(self.sample_file_list[index])
        root,ext=os.path.splitext(a)

        if ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif ext == '.npy':
            points = np.load(self.sample_file_list[index])
        elif ext == '.ply':
            #points = open3d.io.read_point_cloud('/data/taoliu/taoliufile/point_detection/OpenPCDet/data/cylinder/training/points/0001.ply', format='auto', remove_nan_points=False, remove_infinite_points=False, print_progress=False)
            print(os.path.exists(a))
            points = open3d.io.read_point_cloud(a, format='auto')
            points = np.asarray(points.points)


        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/work/tl2lab/tl_folder/lidar_work/pcdet_tree2/OpenPCDet/tools/cfgs/cylinder_models/pointpillar_stem.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default="/work/tl2lab/Preprocessing_Output_2/point_slice_006/slice_0.ply",
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='/work/tl2lab/tl_folder/lidar_work/pcdet_tree2/OpenPCDet/stem/output/stem/Previous_Epochs/checkpoint_epoch_300.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.ply', help='specify the extension of your point cloud data file')
    parser.add_argument('--save_dir', type=str, default='/work/tl2lab/Preprocessing_Output_2/malaysia006', help='directory to save inference outputs')
    parser.add_argument('--vis', type=int, choices=[0, 1], default=0, help='visualization switch: 0=off, 1=on')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    args.no_vis = (args.vis == 0)

    return args, cfg


args, cfg = parse_config()
logger = common_utils.create_logger()
logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
demo_dataset = DemoDataset(
    dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
    root_path=Path(args.data_path), ext=args.ext, logger=logger
)
logger.info(f'Total number of samples: \t{len(demo_dataset)}')

save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
print(save_dir.resolve())
logger.info(f'Saving inference outputs to: {save_dir.resolve()}')

model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
model.cuda()
model.eval()
with torch.no_grad():
    for idx, data_dict in enumerate(demo_dataset):
        logger.info(f'Visualized sample index: \t{idx + 1}')
        data_dict = demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        pred_dicts, _ = model.forward(data_dict)
        sample_points = data_dict['points'][:, 1:].cpu().numpy()
        pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
        pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
        pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()

        sample_name = Path(demo_dataset.sample_file_list[idx]).stem
        output_path = save_dir / f'{idx:06d}_{sample_name}.npz'
        np.savez_compressed(
            output_path,
            points=sample_points,
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_labels=pred_labels,
        )
        logger.info(f'Saved: {output_path}')

        if not args.no_vis:
            V.draw_scenes(
                points=sample_points, ref_boxes=pred_boxes,
                ref_scores=pred_scores, ref_labels=pred_labels
            )

        #mlab.show(stop=True)

logger.info('Demo done.')




# %%
