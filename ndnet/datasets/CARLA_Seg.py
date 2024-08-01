import torch
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
import os
from typing import Tuple, List
from ndnet.preprocessing.ndt_legacy import NDT_Sampler

class CARLA_Seg(Dataset):
    """
    CARLA semantic segmentation dataset.

    Args:
        n_classes (int): number of point classes. Don't count with unlabeled/unknown class.
        n_samples (int): number of points to sample (unsing farthest point sampling)
        path (str): path to the dataset
    """
    def __init__(self, n_classes: int, n_samples: int, path: str) -> None:
        super().__init__()

        self.n_classes: int = n_classes
        self.n_samples = n_samples
        self.path: str = path

        # verify that the path exists
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset not found at {self.path}")
        
        # get the list of files in the dataset
        self.filenames: List[str] = os.listdir(self.path)
        # sort the filenames
        self.filenames.sort()
        

    def __len__(self) -> int:
        # return the length of the filenames list
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample composed by a point cloud and its segmentation ground truth.

        Args:
            idx (int): index of the sample
        """

        # verify that the index is within the limits
        if idx < 0 or idx >= len(self.filenames):
            raise IndexError(f"Index {idx} out of bounds")

        # retrieve the filename of the sample
        pcl_filename: str = os.path.join(self.path, self.filenames[idx])

        pcl, gt = self.get_data_pcl(pcl_filename)

        return pcl, gt
            
        
    def color_to_class(self, color: np.array) -> int:
        """
        Convert a color to a class tag.

        Args:
            color (Tuple[int, int, int]): RGB color. Values are in range [0, 1]

        Returns:
            int: class tag
        """

        # scale the color values to the range [0, 255]
        color = (color * 255).astype(np.uint8)

        # convert the RGB color to a single integer
        color_int = color[0] << 16 | color[1] << 8 | color[2]

        return color_int
    
    def class_to_color(self, class_tag: int) -> np.array:
        """
        
        Convert a class tag to a color.

        Args:
            class_tag (int): class tag

        Returns:
            Tuple[int, int, int]: RGB color. Values are in range [0, 1]
        """

        # get the red, green and blue components
        r = (class_tag >> 16) & 0xff
        g = (class_tag >> 8) & 0xff
        b = class_tag & 0xff

        return np.array([r, g, b], dtype=np.float32) / 255.0
        
    def get_data_pcl(self, pcl_filename: str, num_header_lines: int = 10) -> Tuple[torch.Tensor, ]:
        """
        Get the data from a given PLY file.

        Args:
            pcl_filename (str): path to the PLY file
            num_header_lines (int): number of header lines in the PLY file (default: 10)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: point cloud and segmentation ground truth tensors
        """

        points: List[List[float]] = []  # Points coordinates
        classes: List[int] = []  # Class tag of the points
        n_points = 0                # Number of points in the point cloud

        # open the file descriptor and read file lines
        pcl = None
        with open(pcl_filename, 'r') as f:
            pcl = f.readlines()
        f.close()

        # process each line after the header
        for point in pcl[num_header_lines:]:

            data: List[int] = point.strip().split() # split the line to get each coordinate
            x = float(data[0])
            y = float(data[1])
            z = float(data[2])
            class_tag = int(data[-1]) # the class tag is the last element

            if class_tag > self.n_classes:
                raise ValueError(f"Class tag {class_tag} out of bounds")

            # append the point coordinates to the points list
            points.append(np.array([x, y, z]))

            # append the class to the classes list
            classes.append(class_tag)

            n_points += 1

        np_points = np.asarray(points)

        # randomly select points
        point_indexes = np.random.choice(np_points.shape[0], self.n_samples, replace=False)
        np_points = np_points[point_indexes]

        # create numpy arrays with the points
        # np_points = np.asarray(points)
        np_classes = np.asarray(classes, dtype=np.uint16)
        np_classes = np_classes[point_indexes]

        """
        # create the Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_points)
        # assign the colors to the point cloud
        pcd.colors = o3d.utility.Vector3dVector([self.class_to_color(c) for c in np_classes])
        # downsample using FPS
        # pcd = pcd.farthest_point_down_sample(self.n_samples)

        # create a tensor from the points
        # points = torch.tensor(np.asarray(pcd.points)).float()
        points = torch.tensor(np_points).float()

        # create a list of classes from the colors
        np_colors = np.asarray(pcd.colors)
        np_classes = np.array([self.color_to_class(c) for c in np_colors])
        """

        points = torch.tensor(np_points).float()
        
        # make the ground truth tensor with one-hot encoding
        gt = torch.zeros((np_classes.shape[0], self.n_classes+1)).float()
        for i in range(np_classes.shape[0]):
            gt[i, int(np_classes[i])] = 1

        return points, gt