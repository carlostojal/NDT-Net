import torch
from torch.utils.data import Dataset
import numpy as np
import os
from typing import Tuple, List

class CARLA_Seg(Dataset):
    """
    CARLA semantic segmentation dataset.

    Args:
        n_classes (int): number of point classes
        path (str): path to the dataset
    """
    def __init__(self, n_classes: int, path: str) -> None:
        super().__init__()

        self.n_classes: int = n_classes
        self.path: str = path

        # verify that the path exists
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset not found at {self.train_path}")
        
        # get the list of files in the dataset
        self.filenames: List[str] = os.listdir(self.path)
        

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
        pcl_filename: str = self.filenames[idx]

        # get the point cloud and the segmentation ground truth
        pcl, gt = self.get_data_pcl(pcl_filename)

        return pcl, gt
        
    def get_data_pcl(self, pcl_filename: str, num_header_lines: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the data from a given PLY file.

        Args:
            pcl_filename (str): path to the PLY file
            num_header_lines (int): number of header lines in the PLY file (default: 10)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: point cloud and segmentation ground truth tensors
        """

        points: List[List[float]] = [] # Points coordinates
        classes: List[List[int]] = [] # Class tag of the points
        n_points = 0                # Number of points in the point cloud

        # open the file descriptor and read file lines
        pcl = None
        with open(pcl_filename, 'r') as f:
            pcl = f.readlines()

        # process each line
        for point in pcl[num_header_lines:]:

            class_cur: np.ndarray = np.zeros((self.n_classes)) # create a matrix for the class distribution of this point

            data: List[int] = point.strip().split() # split the line to get each coordinate
            x = float(data[0])
            y = float(data[1])
            z = float(data[2])

            # append the point coordinates to the points list
            points.append([x, y, z])

            # append the class distribution to the classes list
            classes.append(class_cur)

            n_points += 1

        # return the matrices as tensors
        return torch.tensor(np.asarray(points)), torch.tensor(np.asarray(classes))       
