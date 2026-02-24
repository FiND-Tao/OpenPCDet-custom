#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import open3d as o3d
import os

def combine_npy_files_from_folder(folder_path, output_file):
    file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')]
    combined_array = []

    for file in file_list:
        array = np.load(file)
        combined_array.append(array)
    
    combined_array = np.concatenate(combined_array, axis=0)
    np.save(output_file, combined_array)
    return combined_array

def convert_to_mesh(combined_array, output_mesh_file):
    # Here I create a directory to store the mesh file
    output_dir = os.path.dirname(output_mesh_file)
    os.makedirs(output_dir, exist_ok=True)

    # Creating lists to store vertices and faces of all bounding boxes
    all_vertices = []
    all_faces = []

    # Here I generate vertices and faces for each bounding box
    for i in range(combined_array.shape[0]):
        center = combined_array[i, 0:3]
        lwh = combined_array[i, 3:6]
        axis_angles = np.array([0, 0, combined_array[i, 6] + 1e-10])
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
        bbox = o3d.geometry.OrientedBoundingBox(center, rot, lwh)
        vertices = np.asarray(bbox.get_box_points())
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 5], [0, 4, 5], [0, 3, 7], [0, 4, 7], [6, 2, 3], [6, 2, 1], [6, 7, 3], [6, 7, 4], [6, 5, 1], [6, 5, 4]])
        
        # Adding the vertices and faces to the lists
        all_vertices.append(vertices)
        all_faces.append(faces)

    # Merging the vertices and faces are done here
    merged_vertices = np.concatenate(all_vertices)
    offset = 0
    for faces in all_faces:
        faces += offset
        offset += len(all_vertices[0])

    merged_faces = np.concatenate(all_faces)

    # Here is where I create a mesh for the combined bounding boxes (but now its not needed, can remove later)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(merged_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(merged_faces)

    # Save to ply file
    o3d.io.write_triangle_mesh(output_mesh_file, mesh)
    print(f'Mesh saved to {output_mesh_file}')

folder_path = r'G:\Andys_Proposal\3 point cloud\Geoslam_Aerial'
combined_npy_file = r"G:\Andys_Proposal\3 point cloud\Geoslam_Aerial\Geoslam_Aerial_CS.npy"
combined_array = combine_npy_files_from_folder(folder_path, combined_npy_file)

output_mesh_file = r'G:\Andys_Proposal\3 point cloud\Geoslam_Aerial\Geoslam_Aerial_Mesh.ply'
convert_to_mesh(combined_array, output_mesh_file)


# In[ ]:




