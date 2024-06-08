import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple
from ..preprocessing.ndt_legacy import NDT_Sampler
from .ndtnet import NDTNet

class ResidualConnection(nn.Module):
    """
    Change the number of feature vectors, either up or down.
    """
    def __init__(self, in_points: int, out_points: int) -> None:
        super().__init__()

        self.in_points = in_points
        self.out_points = out_points

        self.conv1 = nn.Conv1d(in_points, out_points, 1)
        self.bn1 = nn.BatchNorm1d(out_points)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual connection.

        Args:
        - x (torch.Tensor): the input tensor, shaped (batch_size, in_points, feature_dim)

        Returns:
        - torch.Tensor: the output tensor, shaped (batch_size, out_points, feature_dim)
        """
        
        # transpose the tensor
        x = x.transpose(1, 2) # [B, N, F]

        # apply the convolution, batch normalization and ReLU
        x = F.relu(self.bn1(self.conv1(x)))

        # transpose back
        x = x.transpose(1, 2) # [B, F, N]

        return x


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual connection.

        Args:
        - x (torch.Tensor): the input tensor, shaped (batch_size, in_dim, num_points)

        Returns:
        - torch.Tensor: the output tensor, shaped (batch_size, out_dim, num_points)
        """

class NDTNetpp(nn.Module):
    def __init__(self, point_dim: int = 3,
                 fine_res: int = 8160, coarse_res: int = 4080, 
                 feature_dim: int = 1024) -> None:
        super().__init__()

        self.point_dim = point_dim

        # receive the resolutions
        self.coarse_res = coarse_res
        self.fine_res = fine_res

        # receive the feature dimension
        self.feature_dim = feature_dim

        # ndt-net layers
        self.ndtnet1 = NDTNet(point_dim=point_dim, feature_dim=feature_dim, extra_type=NDTNet.AdditionalFeatures.COVARIANCES)
        self.ndtnet2 = NDTNet(point_dim=point_dim, feature_dim=feature_dim, extra_type=NDTNet.AdditionalFeatures.FEATURE_VECTOR)

        # residual connection
        self.residual = ResidualConnection(self.fine_res, self.coarse_res)

        # last conv layer
        self.conv1 = nn.Conv1d(feature_dim, feature_dim, 1)

        # batch normalization
        self.bn1 = nn.BatchNorm1d(feature_dim)

        # get the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def forward(self, points1: torch.Tensor, covariances1: torch.Tensor, sampler1: NDT_Sampler,
                points2: torch.Tensor, covariances2: torch.Tensor, sampler2: NDT_Sampler) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the NDT-Net++ feature extractor.

        Args:
        - points (torch.Tensor): the points tensor, shaped (batch_size, num_points, point_dim)
        - covariances (torch.Tensor): the covariances tensor, shaped (batch_size, num_points, covariance_dim)

        Returns:
        - torch.Tensor: tensor with shape (batch_size, 1024, num_points)
        """

        if sampler1 is None or sampler2 is None:
            raise ValueError("NDT-Net++ requires two NDT samplers")

        # BRANCH 1

        # ndt-net layer on the finer resolution
        feat1, _ = self.ndtnet1(points1, covariances1) # [B, 1024, N1]

        # finer resolution pruning to the number of points in the coarser resolution
        down1, downcov1, _ = sampler1.prune(self.coarse_res)
        down1 = down1.to(self.device)
        downcov1 = downcov1.to(self.device)

        # residual connection to reduce the number of feature vectors
        feat1_ = self.residual(feat1) # [B, 1024, N2]
        # concat downsampled points, covariances and feature vectors
        downcov1 = torch.cat([down1, downcov1, feat1_], dim=1) # [B, 1033, N2]

        # second ndt-net layer, on the downsampled points and features
        feat1_ = self.ndtnet2(down1, downcov1) # [B, 1024, N2]

        # BRANCH 2

        # ndt-net layer on the coarser resolution
        feat2 = self.ndtnet2(points2, covariances2) # [B, 1024, N2]

        # MERGE THE BRANCHES

        # sum the features
        feat_ = feat1_ + feat2

        # last conv layer
        feat = self.bn1(self.conv1(feat_)) # [B, 1024, N2]

        return feat, feat1
    
class NDTNetppClassification(nn.Module):

    def __init__(self, point_dim: int = 3, num_classes: int = 512, 
                 fine_res: int = 8160,
                 coarse_res: int = 4080,
                 feature_dim: int = 1024) -> None:

        super().__init()

        self.point_dim = point_dim
        self.num_classes = num_classes

        self.fine_res = fine_res
        self.coarse_res = coarse_res
        self.feature_dim = feature_dim

        self.feature_extractor = NDTNetpp(point_dim, fine_res, coarse_res, feature_dim)

        self.conv1 = nn.Conv1d(feature_dim, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, num_classes, 1)

    def forward(self, points1: torch.Tensor, covariances1: torch.Tensor, sampler1: NDT_Sampler,
                points2: torch.Tensor, covariances2: torch.Tensor, sampler2: NDT_Sampler) -> torch.Tensor:

        if sampler1 is None or sampler2 is None:
            raise ValueError("NDT-Net++ requires two NDT samplers")
        
        # extract features
        x, _ = self.feature_extractor(points1, covariances1, sampler1, points2, covariances2, sampler2)

        # max pooling
        x = torch.max(x, 2, keepdim=True)[0]

        # FC layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)

        # softmax
        x = torch.softmax(x, dim=1)

        return x

class NDTNetppSegmentation(nn.Module):

    def __init__(self, point_dim: int = 3, 
                 num_classes: int = 16,
                 fine_res: int = 8160,
                 coarse_res: int = 4080,
                 feature_dim: int = 1024) -> None:
        
        super().__init__()
        
        self.point_dim = point_dim

        # receive the resolutions
        self.coarse_res = coarse_res
        self.fine_res = fine_res

        # receive the feature dimension
        self.feature_dim = feature_dim

        # initialize NDT-Net++ feature extractor
        self.ndtnetpp = NDTNetpp(point_dim, fine_res, coarse_res, feature_dim)

        # residual connection to upsample back to the finer resolution
        self.residual = ResidualConnection(coarse_res, fine_res)

        # initialize the segmentation head
        self.conv1 = nn.Conv1d(self.feature_dim, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes+1, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, points1: torch.Tensor, covariances1: torch.Tensor, sampler1: NDT_Sampler,
                points2: torch.Tensor, covariances2: torch.Tensor, sampler2: NDT_Sampler) -> torch.Tensor:

        if sampler1 is None or sampler2 is None:
            raise ValueError("NDT-Net++ requires two NDT samplers")
        
        # extract features. shapes [B, F, N2] and [B, F, N1]
        x, x1 = self.ndtnetpp(points1, covariances1, sampler1, points2, covariances2, sampler2)

        # upsample the features
        x = self.residual(x) # [B, F, N1]

        # sum the features
        x = x + x1 # [B, F, N1]

        # conv layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # softmax
        x = F.softmax(self.conv4(x), dim=1)

        x = x.transpose(1, 2) # [B, N1, C]

        return x
