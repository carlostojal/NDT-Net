import torch
from torch.utils.data import Dataset
import numpy as np
import os
from typing import Tuple, List
from ndtnetpp.preprocessing.ndt_legacy import NDT_Sampler

class CARLA_NDT_Seg(Dataset):
    """
    CARLA NDT semantic segmentation dataset.

    Args:
        n_classes (int): number of point classes. Don't count with unlabeled/unknown class.
        n_desired_nds (int): number of desired normal distributions
        path (str): path to the dataset
    """
    def __init__(self, n_classes: int, n_desired_nds: int, path: str) -> None:
        super().__init__()

        self.n_classes: int = n_classes
        self.n_desired_nds: int = n_desired_nds
        self.path: str = path

        # verify that the path exists
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset not found at {self.path}")
        
        # get the list of files in the dataset
        self.filenames: List[str] = os.listdir(self.path)
        

    def __len__(self) -> int:
        # return the length of the filenames list
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # get the point cloud and the segmentation ground truth
        pcl, cov, gt = self.get_data_pcl(pcl_filename)

        return pcl, cov, gt
        
    def get_data_pcl(self, pcl_filename: str, num_header_lines: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the data from a given PLY file.

        Args:
            pcl_filename (str): path to the PLY file
            num_header_lines (int): number of header lines in the PLY file (default: 10)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: point cloud, covariances and segmentation ground truth tensors
        """

        points: List[List[float]] = []  # Points coordinates
        classes: List[int] = []  # Class tag of the points
        n_points = 0                # Number of points in the point cloud

        # open the file descriptor and read file lines
        pcl = None
        with open(pcl_filename, 'r') as f:
            pcl = f.readlines()
        f.close()

        # process each line
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
        np_classes = np.asarray(classes, dtype=np.uint16)

        # sample using NDT
        sampler: NDT_Sampler = NDT_Sampler(np_points, np_classes, self.n_classes)
        np_points1, np_covariances1, np_classes1 = sampler.downsample(self.n_desired_nds)
        sampler.cleanup()

        points = torch.tensor(np_points1).float()
        covariances = torch.tensor(np_covariances1).float()
        classes = torch.tensor(np_classes1).float()

        # replace NaN values with zeros
        points[torch.isnan(points)] = 0
        covariances[torch.isnan(covariances)] = 0

        # make the ground truth tensor with one-hot encoding
        gt = torch.zeros((classes.shape[0], self.n_classes+1)).float()
        for i in range(classes.shape[0]):
            gt[i, int(classes[i])] = 1

        # points: [n_points, 3]
        # covariances: [n_points, 9]
        # gt: [n_points, n_classes+1]

        # return the tensors
        return points, covariances, gt
