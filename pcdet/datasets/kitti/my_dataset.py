# Import necessary libraries
import glob
import numpy as np
from pathlib import Path
from ..dataset import DatasetTemplate
import open3d as o3d
# Define the MyDataset class that inherits from DatasetTemplate
class MyDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Initialize the MyDataset instance.
        
        Args:
            dataset_cfg: Configuration parameters for the dataset.
            class_names: List of class names for the dataset.
            training: Boolean flag indicating if the dataset is used for training.
            root_path: The root directory where the dataset is stored.
            logger: A logger instance for logging information.
        """
        # Initialize the base class with provided arguments
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        
        # Find all point cloud files (.bin) and label files (.txt) in the dataset's 'training' folder and sort them
        if training:
            print(str(self.root_path / 'training/points/*.ply'))
            point_file_list = glob.glob(str(self.root_path / 'training/points/*.ply'))
            labels_file_list = glob.glob(str(self.root_path / 'training/labels/*.txt'))
            point_file_list.sort()
            labels_file_list.sort()
        else:
            print(str(self.root_path / 'testing/points/*.ply'))
            point_file_list = glob.glob(str(self.root_path / 'testing/points/*.ply'))
            labels_file_list = glob.glob(str(self.root_path / 'testing/labels/*.txt'))
            point_file_list.sort()
            labels_file_list.sort()
        # Store the sorted file lists in instance variables
        self.sample_file_list = point_file_list
        self.samplelabel_file_list = labels_file_list

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.sample_file_list)

    def __getitem__(self, index):
        # Retrieves a data sample given an index
        
        # Get the file stem (filename without the path and extension)
        sample_idx = Path(self.sample_file_list[index]).stem
        
        # Load point cloud data from a binary file and reshape it to have 3 columns (assuming x, y, z)
        #points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)[:, :3]
        pointsobject = o3d.io.read_point_cloud(self.sample_file_list[index])
        points=np.asarray(pointsobject.points)
        # Transform the coordinates of the points (assuming this is required for a coordinate system transformation)
        #points = points[:, [2, 0, 1]]
        #points[:, 0] = -points[:, 0]
        #points[:, 1] = -points[:, 1]
        
        # Load label data, reshape it, and adjust the columns to match the points' coordinate system
        points_label = np.loadtxt(self.samplelabel_file_list[index],delimiter=',', dtype=np.float32).reshape(-1, 7)
        #points_label = points_label[:, [2, 0, 1, 5, 3, 4, 6]]
        
        # Create an array with the name 'cylinder' for each label (this is placeholder logic and should be adapted)
        gt_names = np.array(['cylinder'] * points_label.shape[0])
        
        # Prepare the input dictionary with the data points, frame id, ground truth names, and boxes
        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            'gt_names': gt_names,
            'gt_boxes': points_label
        }
        
        # Prepare the data using the prepare_data method from DatasetTemplate
        data_dict = self.prepare_data(data_dict=input_dict)
        
        # Return the prepared data dictionary
        return data_dict
    def evaluation(self, det_annos, class_names, **kwargs):
        self.logger.warning('Evaluation is not implemented for Pandaset as there is no official one. ' +
                            'Returning an empty evaluation result.')
        ap_result_str = ''
        ap_dict = {}

        return ap_result_str, ap_dict
