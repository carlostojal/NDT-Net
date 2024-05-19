"""
Visualize NDT output on a point cloud.
"""

import sys
sys.path.append(".")
import argparse
import open3d as o3d
from ndtnetpp.preprocessing.ndt import ndt_downsample

if __name__ == "__main__":

    # initialize argument parser
    parser = argparse.ArgumentParser(description="Visualize NDT output on a point cloud.")
    parser.add_argument("input", help="Input file containing NDT output.")
    args = parser.parse_args()

    # read input file using open3d
    pcd = o3d.io.read_point_cloud(args.input)

    # visualize point cloud
    o3d.visualization.draw_geometries([pcd])

    # downsample the point cloud
    downsampled_pcd = ndt_downsample(pcd, 1000)

    # visualize downsampled point cloud
    o3d.visualization.draw_geometries([downsampled_pcd])    
    