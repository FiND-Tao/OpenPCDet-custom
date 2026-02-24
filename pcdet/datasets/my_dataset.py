from pathlib import Path
import numpy as np
from pcdet.datasets.dataset import DatasetTemplate
import open3d as o3d


class MyDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)

        self.root_path = Path(dataset_cfg.DATA_PATH)
        split = dataset_cfg.DATA_SPLIT['train'] if training else dataset_cfg.DATA_SPLIT['test']

        self.points_dir = self.root_path / split / 'points'
        self.label_dir = self.root_path / split / 'label'

        self.sample_ids = sorted([p.stem for p in self.points_dir.glob('*.ply')])

        self.points_files = [self.points_dir / f'{i}.ply' for i in self.sample_ids]
        self.label_files = [self.label_dir / f'{i}.txt' for i in self.sample_ids]

        assert len(self.points_files) == len(self.label_files), \
            f'Mismatch: {len(self.points_files)} points vs {len(self.label_files)} labels'

        print(f"[MyDataset] root_path = {self.root_path}")
        print(f"[MyDataset] split = {split}")
        print(f"[MyDataset] #samples = {len(self.sample_ids)}")
        print(f"[MyDataset] first point file = {self.points_files[0] if self.sample_ids else 'NONE'}")
        print(f"[MyDataset] first label file = {self.label_files[0] if self.sample_ids else 'NONE'}")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        # Load point cloud
        pcd = o3d.io.read_point_cloud(str(self.points_files[index]))
        points = np.asarray(pcd.points, dtype=np.float32)

        # Load labels (CSV)
        gt_boxes = np.loadtxt(
        self.label_files[index],
        delimiter=',',
        dtype=np.float32
        ).reshape(-1, 7)

        # Remove invalid boxes
        valid_mask = (gt_boxes[:, 3] > 0) & (gt_boxes[:, 4] > 0) & (gt_boxes[:, 5] > 0)
        gt_boxes = gt_boxes[valid_mask]

        # Assign class names
        gt_names = np.array(['cylinder'] * gt_boxes.shape[0])


        input_dict = {
            'points': points,
            'gt_boxes': gt_boxes,
            'gt_names': gt_names,
            'frame_id': self.sample_ids[index]
        }

        return self.prepare_data(input_dict)