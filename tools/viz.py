"""
Visualize NDT output on a point cloud.
"""

import sys
sys.path.append(".")
import argparse
import open3d as o3d
import time
from ndtnetpp.preprocessing.ndt import ndt_downsample

if __name__ == "__main__":

    # initialize argument parser
    parser = argparse.ArgumentParser(description="Visualize NDT output on a point cloud.")
    parser.add_argument("input", help="Input file containing NDT output.")
    args = parser.parse_args()

    # read input file using open3d
    pcd = o3d.io.read_point_cloud(args.input)
    pcd.paint_uniform_color([1, 0, 0])

    print("Point cloud has", len(pcd.points), "points.")

    # create the visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # visualize point cloud
    vis.add_geometry(pcd)

    # downsample the point cloud
    # measure starting time
    start = time.time()
    downsampled_pcd = ndt_downsample(pcd, 1000)
    # measure ending time
    end = time.time()
    duration = end - start
    hz = 1 / duration
    print(f"Downsampling took {duration} seconds - {hz} Hz")
    downsampled_pcd.paint_uniform_color([0, 1, 0])

    # visualize downsampled point cloud
    vis.add_geometry(downsampled_pcd)

    # run the visualizer
    vis.run()

    # destroy the visualizer
    vis.destroy_window()
