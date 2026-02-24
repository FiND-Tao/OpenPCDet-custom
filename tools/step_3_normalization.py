import rasterio
import numpy as np
from pyntcloud import PyntCloud

# Load the point cloud
cloud = PyntCloud.from_file(r'/data/taoliu/Data_Share/CodeSteps/Data/subset_no_outlier.ply')

# Load the DTM using rasterio
with rasterio.open(r"/data/taoliu/Data_Share/CodeSteps/Data/subset_DTMsmooth.tif") as src:
    # Transform the point cloud x,y coordinates to raster indices
    rows, cols = src.index(cloud.points.x.values, cloud.points.y.values)

    # Interpolate the DTM values for each point in the point cloud using NumPy's advanced indexing
    z_dtm = src.read(1)[rows, cols]

# Normalize the point cloud z-values using vectorized operations
cloud.points["z"] -= z_dtm

# Save the normalized point cloud
cloud.to_file(r"/data/taoliu/Data_Share/CodeSteps/Data/point_cloud_normalized.ply")