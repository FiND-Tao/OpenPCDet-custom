#!/usr/bin/env python
# coding: utf-8

# In[1]:


import open3d as o3d
import numpy as np
import os

# File paths
original_pcd_file = r"G:\Andys_Proposal\Parth_Aerial\Andys_Aerial_slice_0.ply"
bounding_box_mesh_file = r"G:\Andys_Proposal\3 point cloud\Geoslam_Aerial\Geoslam_Aerial_Mesh.ply"
output_dir = r"G:\Andys_Proposal\3 point cloud\Geoslam_Aerial\Aerial_Predicted_Stems_clip"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load original point cloud
original_pcd = o3d.io.read_point_cloud(original_pcd_file)

# Load bounding box mesh
mesh = o3d.io.read_triangle_mesh(bounding_box_mesh_file)

# Extract vertices representing bounding boxes
bounding_boxes = np.asarray(mesh.vertices).reshape(-1, 8, 3)  # Each box has 8 corner points

# Process each bounding box separately
for i, box_vertices in enumerate(bounding_boxes):
    # Create an oriented bounding box from the 8 corner points
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(box_vertices))

    # Crop the original point cloud using the bounding box
    cropped_pcd = original_pcd.crop(obb)

    # Save the extracted stem as a separate PLY file
    output_file = os.path.join(output_dir, f"stem_{i}.ply")
    o3d.io.write_point_cloud(output_file, cropped_pcd)

    print(f"Saved: {output_file}")

print("All stems have been extracted and saved.")


# In[ ]:




