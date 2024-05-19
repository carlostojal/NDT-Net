import numpy as np
import open3d as o3d
import ctypes

# import the core shared library
core = ctypes.cdll.LoadLibrary('core/build/libndtnetpp.so')

# downsample with NDT
def ndt_downsample(pointcloud: o3d.geometry.PointCloud, num_desired_points: int) -> o3d.geometry.PointCloud:

    # get the point count
    num_points = len(pointcloud.points)
    pointcloud = np.asarray(pointcloud.points)
    pcl_ptr = pointcloud.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    new_pcl = np.zeros((num_desired_points, 3))
    new_pcl_ptr = new_pcl.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    num_downsampled_points = ctypes.c_int(0)

    # downsample the point cloud
    core.ndt_downsample(pcl_ptr, 3, num_points, num_desired_points, 
                        new_pcl_ptr, num_downsampled_points)
    
    # create a pointcloud from the new_pcl array
    downsampled_pointcloud = o3d.geometry.PointCloud()
    downsampled_pointcloud.points = o3d.utility.Vector3dVector(new_pcl)

    return downsampled_pointcloud
