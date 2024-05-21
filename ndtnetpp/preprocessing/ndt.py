import numpy as np
import open3d as o3d
import ctypes

# import the core shared library
core = ctypes.cdll.LoadLibrary('core/build/libndtnetpp.so')

# downsample with NDT
def ndt_downsample(pointcloud: np.ndarray, num_desired_points: int, classes: np.ndarray = None, num_classes: int = None) -> tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray]:

    # get the point count
    num_points = len(pointcloud)
    pcl_ptr = pointcloud.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # create a new point cloud array
    new_pcl = np.zeros((num_desired_points, 3))
    new_pcl_ptr = new_pcl.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # create pointers to store the number of downsampled points
    num_downsampled_points = ctypes.pointer(ctypes.c_int(0))

    # create a pointer to the classes array
    classes_ptr = None
    if classes is not None:
        classes_ptr = classes.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))

    new_classes = np.zeros(num_desired_points, dtype=np.int16)
    new_classes_ptr = new_classes.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))

    # create a pointer for the covariance
    covariances = np.zeros((num_desired_points, 9), dtype=np.float64)
    covariances_ptr = covariances.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # downsample the point cloud
    core.ndt_downsample(pcl_ptr, 3, num_points,
                        classes_ptr, num_classes,
                        num_desired_points, 
                        new_pcl_ptr, num_downsampled_points,
                        covariances_ptr,
                        new_classes_ptr)
    
    # print the number of downsampled points
    print(f"Number of downsampled points: {num_downsampled_points.contents.value}")
    
    # create a pointcloud from the new_pcl array
    downsampled_pointcloud = o3d.geometry.PointCloud()
    downsampled_pointcloud.points = o3d.utility.Vector3dVector(new_pcl)

    # crop the covariances array
    covariances = covariances[:num_downsampled_points.contents.value]

    return downsampled_pointcloud, covariances, new_classes
