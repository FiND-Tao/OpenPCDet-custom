#%%
#This code only works in the command line
import open3d as o3d
import numpy as np
import os
import pandas as pd
file="/data/taoliu/Sri/point_detection/OpenPCDet/data/stem/training/points/UK sample plot6_slice_0_15x15_Grid_1.ply"
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
label="/data/taoliu/Sri/point_detection/OpenPCDet/data/stem/training/labels/UK sample plot6_slice_0_15x15_Grid_1.txt"
gt_boxes=np.loadtxt(label,delimiter=',')
for i in range(gt_boxes.shape[0]):
    center = gt_boxes[i,0:3]
    lwh = gt_boxes[i,3:6]
    axis_angles = np.array([0, 0, gt_boxes[i,6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)
    boxes.append(box3d)
    vis.add_geometry(box3d)


render_option = vis.get_render_option()
render_option.point_size = 1.0  # Adjust the point size as needed
vis.run()
