import torch
from typing import Tuple
import numpy as np
from ndtnetpp.preprocessing.ndt_legacy import NDT_Sampler

def ndt_preprocessing(num_nds: int, points: torch.Tensor, classes: torch.Tensor = None, num_classes: int = None)-> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess the point cloud to be used in the NDTNet.

    Args:
        points (torch.Tensor): point cloud to be preprocessed (batch_size, num_points, 3)
        classes (torch.Tensor): classes of the point cloud (batch_size, num_points, num_classes)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: normal distribution centers and its classes
    """

    # create the empty tensors
    points_new = torch.empty((points.shape[0], num_nds, 3), dtype=torch.float32).to(points.device)
    covs_new = torch.empty((points.shape[0], num_nds, 9), dtype=torch.float32).to(points.device)
    classes_new = torch.empty((points.shape[0], num_nds, num_classes+1), dtype=torch.float32).to(points.device)

    # iterate the batch dimension
    for b in range(points.shape[0]):

        # convert the points tensor to a numpy array
        points_np = points[b].cpu().numpy().astype(np.float64)

        # convert classes tensor from one-hot encoding to class tags only and then to a numpy array
        if classes is not None:
            classes_np = torch.argmax(classes[b], dim=1).cpu().numpy().astype(np.uint16)
        else:
            classes_np = None

        # create the NDT sampler
        sampler = NDT_Sampler(points_np, classes_np, num_classes)

        # downsample the point cloud
        points_new_np, covs_new_np, classes_new_np = sampler.downsample(num_nds)

        # convert the numpy arrays to tensors
        points_n = torch.from_numpy(points_new_np).to(points.device)
        covs_n = torch.from_numpy(covs_new_np).to(points.device)
        classes_n = torch.from_numpy(classes_new_np).to(points.device)

        # convert the classes to one-hot encoding
        classes_oh = torch.zeros((num_nds, num_classes+1)).float()
        for ndi in range(num_nds):
            classes_oh[ndi, int(classes_n[ndi])] = 1

        # destroy the sampler
        sampler.cleanup()

        # add the new tensors to the batch
        points_new[b] = points_n
        covs_new[b] = covs_n
        classes_new[b] = classes_oh

    # replace nan values with zeros
    points_new = torch.nan_to_num(points_new, nan=0.0, posinf=0.0, neginf=0.0)
    covs_new = torch.nan_to_num(covs_new, nan=0.0, posinf=0.0, neginf=0.0)
    classes_new = torch.nan_to_num(classes_new, nan=0.0, posinf=0.0, neginf=0.0)

    return points_new, covs_new, classes_new
