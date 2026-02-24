#%%
#This code only works in the command line
import open3d as o3d
import numpy as np
import os
import pandas as pd
file="/data/taoliu/Sri/point_detection/OpenPCDet/data/stem/training/points/Brazil_SmallPlots_01_slice_1N_15x15_Grid_5.ply"
#file='/data/taoliu/taoliufile/point_detection/OpenPCDet/data/kitti/testing/velodyne_reduced/000000.bin'
root,ext=os.path.splitext(file)
if ext == '.ply':
    points = o3d.io.read_point_cloud(file, format='auto')
    points = np.asarray(points.points)
elif ext == '.bin':
    points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
    
#gt_boxes=np.load('/data/taoliu/taoliufile/point_detection/OpenPCDet/tools/predbox.npy')

#gt_boxes=np.load('/data/taoliu/taoliufile/point_detection/OpenPCDet/tools/predbox_000079.npy')

boxes=[]
pts = o3d.geometry.PointCloud()
pts.points = o3d.utility.Vector3dVector(points[:, :3])
boxes.append(pts)
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pts)

#label='/data/taoliu/taoliufile/point_detection/OpenPCDet/data/stem/training/labels/0001.txt'
pred_b=np.load('/data/taoliu/Sri/point_detection/OpenPCDet/tools/Brazil_SmallPlots_01_slice_1N_15x15_Grid_5.npy')
label="/data/taoliu/Sri/point_detection/OpenPCDet/data/stem/training/labels/Brazil_SmallPlots_01_slice_1N_15x15_Grid_5.txt"
#gt_boxes=np.loadtxt(label,delimiter=',')
# Set the threshold for confidence score
threshold = 0.5

# Iterate through predictions
for i in range(pred_b.shape[0]):
    center = pred_b[i, 0:3]
    lwh = pred_b[i, 3:6]
    axis_angles = np.array([0, 0, pred_b[i, 6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

    # Calculate pred_scores
    pred_scores = pred_b[i, 0]
    
    # Debugging: Print pred_scores to check values
    print("pred_scores:", pred_scores)

    # Check confidence score against threshold
    if pred_scores >= threshold:
        boxes.append(box3d)
        vis.add_geometry(box3d)


# Adjust point size
render_option = vis.get_render_option()
render_option.point_size = 1.0

# Run visualization
vis.run()
