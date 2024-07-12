import torch
from torch.utils.data import DataLoader
import open3d as o3d
from argparse import ArgumentParser
import numpy as np
import sys
sys.path.append(".")
from ndtnetpp.datasets.CARLA_Seg import CARLA_Seg
from ndtnetpp.models.ndtnet import NDTNetSegmentation
from ndtnetpp.models.pointnet import PointNetSegmentation
from ndtnetpp.preprocessing.ndtnet_preprocessing import ndt_preprocessing

VIZ_SAMPLE = 1

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

if __name__ == '__main__':

    # parse the arguments
    parser = ArgumentParser()
    parser.add_argument("--n_desired_nds", type=int, help="Number of desired normal distributions", default=1000, required=False)
    parser.add_argument("--n_samples", type=int, help="Number of samples to take initially using FPS", default=70000, required=False)
    parser.add_argument("--data_path", type=str, help="Path to the dataset", required=True)
    parser.add_argument("--n_classes", type=int, help="Number of classes. Don't count with unknown/no class", default=28, required=False)
    parser.add_argument("--feature_dim", type=int, help="Dimension of the feature vectors.", default=768, required=False)
    args = parser.parse_args()

    # get the device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    # create the dataset
    print("Creating the dataset...", end=" ")
    dataset = CARLA_Seg(int(args.n_classes), int(args.n_samples), args.train_path)
    print("done.")

    # create the dataloader
    print("Creating the data loader...", end=" ")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    print("done.")

    # create the model
    print("Creating the model...", end=" ")
    model = NDTNetSegmentation(point_dim=3, num_classes=int(args.n_classes), feature_dim=int(args.feature_dim))
    # model = PointNetSegmentation(point_dim=3, num_classes=int(args.n_classes), feature_dim=int(args.feature_dim))
    model = model.to(device)
    print("done.")

    # set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        
        for i, (pcl, gt) in enumerate(dataloader):

            if i == VIZ_SAMPLE:

                # transfer the tensors to the device
                pcl = pcl.to(device)
                gt = gt.to(device)

                # infer the model
                pred = model(pcl)

                # get the predicted classes tensor
                pred_classes = torch.argmax(pred, dim=2)

                # convert to numpy arrays
                points_np = np.asarray(pcl)
                pred_classes_np = np.asarray(pred_classes)

                # create the open3d point cloud object
                pcd_obj = paint_class(points_np, pred_classes_np, int(args.n_classes))

                # visualize the point cloud
                vis = o3d.visualization.Visualizer()
                vis.create_window()
                vis.add_geometry(pcd_obj)
                vis.run()
                vis.destroy_window()
                