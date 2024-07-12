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

VIZ_SAMPLE = 0

def paint_class(pcd: np.ndarray, classes: np.ndarray, num_classes: int = 29) -> o3d.geometry.PointCloud:

    colors = [
        [1, 0, 0], # red
        [0, 1, 0], # green
        [0, 0, 1], # blue
        [1, 1, 0], # yellow
        [1, 0, 1], # magenta
        [0, 1, 1], # cyan
        [1, 0.5, 0], # orange
        [1, 0, 0.5], # pink
        [0.5, 1, 0], # lime
        [0, 1, 0.5], # mint
        [0.5, 0, 1], # purple
        [0, 0.5, 1], # sky
        [0.5, 1, 1], # turquoise
        [1, 0.5, 1], # rose
        [1, 1, 0.5], # banana
        [0.5, 0, 0], # maroon
        [0, 0.5, 0], # olive
        [0, 0, 0.5], # navy
        [0.5, 0.5, 0], # brown
        [0.5, 0, 0.5], # purple
        [0, 0.5, 0.5], # teal
        [0.5, 0.5, 1], # lavender
        [0.5, 1, 0.5], # mint
        [1, 0.5, 0.5], # pink
        [0.5, 0, 0.5], # purple
        [0.5, 0.5, 0.5], # gray
        [0.25, 0.25, 0], # olive
        [0, 0.25, 0.25], # navy
        [0.25, 0, 0.25], # purple
    ]

    # create a point cloud object
    pcd_obj = o3d.geometry.PointCloud()

    # iterate the points
    for i in range(pcd.shape[0]):
        
        # get the class
        class_ = classes[i]

        """
        # get the color. distribute colors evenly across the spectrum in 3 bytes (RGB)
        color = int(2**24 / (num_classes - class_))
        # print(f"Color {color} for class {class_}")
        r = int(color & (0xFF << 16)) / 255.0
        g = int(color & (0xFF << 8)) / 255.0
        b = int(color & 0xFF) / 255.0
        """

        # create the point and color
        pcd_obj.points.append(pcd[i])
        pcd_obj.colors.append(colors[class_])

    return pcd_obj

if __name__ == '__main__':

    # parse the arguments
    parser = ArgumentParser()
    parser.add_argument("--n_desired_nds", type=int, help="Number of desired normal distributions", default=1000, required=False)
    parser.add_argument("--n_samples", type=int, help="Number of samples to take initially using FPS", default=70000, required=False)
    parser.add_argument("--data_path", type=str, help="Path to the dataset", required=True)
    parser.add_argument("--weights_path", type=str, help="Path to the model weights", required=True)
    parser.add_argument("--n_classes", type=int, help="Number of classes. Don't count with unknown/no class", default=28, required=False)
    parser.add_argument("--feature_dim", type=int, help="Dimension of the feature vectors.", default=768, required=False)
    args = parser.parse_args()

    # get the device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create the dataset
    print("Creating the dataset...", end=" ")
    dataset = CARLA_Seg(int(args.n_classes), int(args.n_samples), args.data_path)
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

    # load the model weights
    print("Loading the model weights...", end=" ")
    model.load_state_dict(torch.load(args.weights_path))
    print("done.")

    # set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        
        for i, (pcl, gt) in enumerate(dataloader):

            print(i)

            if i == VIZ_SAMPLE:

                # transfer the tensors to the device
                pcl = pcl.to(device)
                gt = gt.to(device)

                # remove the "1" dimension
                pcl = pcl.squeeze(1)
                gt = gt.squeeze(1) # (batch_size, num_points, num_classes)

                # preprocess the batch
                pcl, covs, gt = ndt_preprocessing(int(args.n_desired_nds), pcl, gt, int(args.n_classes))

                # infer the model
                pred = model(pcl, covs)

                # get the predicted classes tensor
                pred_classes = torch.argmax(pred, dim=2)

                # convert to numpy arrays
                points_np = np.asarray(pcl.squeeze().cpu())
                pred_classes_np = np.asarray(pred_classes.squeeze().cpu())

                # create the open3d point cloud object
                pcd_obj = paint_class(points_np, pred_classes_np, int(args.n_classes))

                # visualize the point cloud
                vis = o3d.visualization.Visualizer()
                vis.create_window()
                vis.add_geometry(pcd_obj)
                vis.run()
                vis.destroy_window()

                break
                