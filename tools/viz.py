"""
Visualize NDT output on a point cloud.
"""

import sys
sys.path.append(".")
import argparse
import open3d as o3d
import numpy as np
import time
from ndtnetpp.preprocessing.ndt_legacy import NDT_Sampler

def read_pcl(file: str, header_lines: int = 10, class_pos: int = 5) -> tuple[np.ndarray,np.ndarray]:
    """
    Read a point cloud from a file.
    """
    # read the file
    lines = None
    with open(file, "r") as f:
        lines = f.readlines()

    # skip the header. the coordinates are the first three columns
    lines = lines[header_lines:]

    # keep the coordinates in an array
    points = []
    classes = []
    for l in lines:

        # split by spaces
        l = l.split()

        # append the coordinates to the points array
        points.append([float(l[0]), float(l[1]), float(l[2])])

        # append the class to the classes array
        classes.append(int(l[class_pos]))

    points = np.array(points, dtype=np.float64)
    classes = np.array(classes, dtype=np.int16)

    return points, classes
    

def paint_class(pcd: np.ndarray, classes: np.ndarray, num_classes: int = 29) -> o3d.geometry.PointCloud:

    # create a point cloud object
    pcd_obj = o3d.geometry.PointCloud()

    # iterate the points
    for i in range(pcd.shape[0]):
        
        # get the class
        class_ = classes[i]

        # get the color. distribute colors evenly across the spectrum in 3 bytes (RGB)
        color = int(2**24 / (num_classes - class_))
        # print(f"Color {color} for class {class_}")
        r = int(color & (0xFF << 16)) / 255.0
        g = int(color & (0xFF << 8)) / 255.0
        b = int(color & 0xFF) / 255.0

        # create the point and color
        pcd_obj.points.append(pcd[i])
        pcd_obj.colors.append([r, g, b])

    return pcd_obj


if __name__ == "__main__":

    # initialize argument parser
    parser = argparse.ArgumentParser(description="Visualize NDT output on a point cloud.")
    parser.add_argument("--input", help="Input file containing point cloud and classes.", type=str, required=True)
    parser.add_argument("--target", help="Target number of points", type=int, default=1000, required=True)
    parser.add_argument("--target1", help="Target number of points", type=int, default=None, required=False)
    parser.add_argument("--classes", help="Number of classes", type=int, default=29, required=False)
    args = parser.parse_args()

    # read input file
    pcd, classes = read_pcl(args.input)

    # instantiate the NDT sampler
    sampler = NDT_Sampler(pcd, classes, int(args.classes))

    # pcd.paint_uniform_color([1, 0, 0])
    pcd_painted = paint_class(pcd, classes, int(args.classes))

    print("Point cloud has", len(pcd), "points.")

    # create the visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # visualize point cloud
    # vis.add_geometry(pcd_painted)

    # downsample the point cloud
    # measure starting time
    start = time.time()
    downsampled_pcd, covariances, classes = sampler.downsample(int(args.target))
    print("Downsampled point cloud has", len(downsampled_pcd), "points.")
    # measure ending time
    end = time.time()
    duration = end - start
    hz = 1 / duration
    print(f"Downsampling took {duration} seconds - {hz} Hz")

    # convert to open3d point cloud object
    downsampled_pcd_obj = o3d.geometry.PointCloud()
    downsampled_pcd_obj.points = o3d.utility.Vector3dVector(downsampled_pcd)
    downsampled_pcd_obj.paint_uniform_color([0, 1, 0])

    # downsampled_pcd_obj = paint_class(np.array(downsampled_pcd.points), classes, int(args.classes))

    # visualize downsampled point cloud
    vis.add_geometry(downsampled_pcd_obj)

    if args.target1 is not None:
        # prune the point cloud
        # measure starting time
        start = time.time()
        pruned_pcd, covariances, classes = sampler.prune(int(args.target1))
        print("Pruned point cloud has", len(pruned_pcd), "points.")
        # measure ending time
        end = time.time()
        duration = end - start
        hz = 1 / duration
        print(f"Pruning took {duration} seconds - {hz} Hz")

        # convert to open3d point cloud object
        pruned_pcd_obj = o3d.geometry.PointCloud()
        pruned_pcd_obj.points = o3d.utility.Vector3dVector(pruned_pcd)
        pruned_pcd_obj.paint_uniform_color([0, 0, 1])

        # visualize pruned point cloud
        vis.add_geometry(pruned_pcd_obj)

    # run the visualizer
    vis.run()

    # destroy the visualizer
    vis.destroy_window()
