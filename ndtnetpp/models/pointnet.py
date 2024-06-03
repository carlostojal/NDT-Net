import torch
from torch import nn
import numpy as np

class TNet(nn.Module):
    """
    Transformation Network
    """

    def __init__(self, in_dim: int = 64) -> None:
        super().__init__()

        self.in_dim = in_dim

        self.conv1 = nn.Conv1d(in_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, in_dim**2)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformation Network

        Args:
        - x (torch.Tensor): the input point cloud, shaped (batch_size, point_dim, num_points)

        Returns:
        - torch.Tensor: the output of the Transformation Network
        """

        # MLP
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x))) # (batch_size, n_points, 1024) shape

        # max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024) # (batch_size, 1024) shape

        # FC layers
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # add identity matrix
        x += torch.eye(self.in_dim).view(1, self.in_dim**2).repeat(x.size(0), 1).to(x.device)
        x = x.view(-1, self.in_dim, self.in_dim)

        return x


class PointNet(nn.Module):
    """
    PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
    """

    def __init__(self, point_dim: int = 3) -> None:
        super().__init__()

        self.point_dim = point_dim
        self.covariance_dim = point_dim**2

        self.conv1 = nn.Conv1d(self.point_dim+self.covariance_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.t1 = TNet(in_dim=point_dim)
        self.t2 = TNet(in_dim=64)


    def forward(self, points: torch.Tensor, covariances: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PointNet

        Args:
        - points (torch.Tensor): the points tensor, shaped (batch_size, num_points, point_dim)
        - covariances (torch.Tensor): the covariances tensor, shaped (batch_size, num_points, covariance_dim)

        Returns:
        - torch.Tensor: the output of the PointNet
        """
        
        # concatenate the covariances to the input tensor, making 12-dimensional points
        x = torch.cat((points, covariances), dim=2)

        B, N, D = x.size()

        x = x.transpose(2, 1) # [B, 12, N]

        # input transform
        p = x[:, :self.point_dim, :] # [B, 3, N]
        t = self.t1(p) # [B, 3, 3]
        # apply the transformation matrix to the points
        p = torch.bmm(t, p) # [B, 3, N]
        p = p.transpose(2, 1) # [B, N, 3]
        # apply the transformation matrix to the covariances
        cov = x[:, self.point_dim:, ] # [B, 9, N]
        # multiply the transition matrix with the covariances
        cov = cov.transpose(2, 1) # [B, N, 9]
        cov = cov.view(B, N, 3, 3) # [B, N, 3, 3]
        cov = torch.matmul(t.unsqueeze(1), cov) # [B, N, 3, 3]
        cov = cov.view(B, N, 9) # [B, N, 9]
        # concatenate the transformed points and covariances
        x = torch.cat((p, cov), dim=2) # [B, N, 12]
        x = x.transpose(2, 1) # [B, 12, N]

        # MLP
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.bn1(self.conv1(x))

        # feature transform
        t = self.t2(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, t)
        x = x.transpose(2, 1)

        x_t2 = x

        # MLP
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))

        # return a tensor with shape (batch_size, 1024, num_points) and the feature transform
        return x, x_t2

class PointNetClassification(nn.Module):

    def __init__(self, point_dim: int = 3, num_classes: int = 512) -> None:
        super().__init__()

        self.point_dim = point_dim
        self.num_classes = num_classes

        self.feature_extractor = PointNet(point_dim)

        self.conv1 = nn.Conv1d(1024, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, num_classes, 1)

    def forward(self, points: torch.Tensor, covariances: torch.Tensor) -> torch.Tensor:
        # extract features
        x, _ = self.feature_extractor(points, covariances)

        # max pooling
        x = torch.max(x, 2, keepdim=True)[0]

        # FC layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)

        # softmax
        x = torch.softmax(x, dim=1)

        return x
    
class PointNetSegmentation(nn.Module):

    def __init__(self, point_dim: int = 3, num_classes: int = 16) -> None:
        super().__init__()

        self.point_dim = point_dim
        self.num_classes = num_classes

        self.feature_extractor = PointNet(point_dim)

        self.conv1 = nn.Conv1d(1088, 512, 1) # 1088 = 1024 + 64
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes+1, 1)

    def forward(self, points: torch.Tensor, covariances: torch.Tensor) -> torch.Tensor:

        # extract features
        x, x_t2 = self.feature_extractor(points, covariances)

        # max pooling
        x, _ = torch.max(x, 2, keepdim=True) # (batch_size, 1024, 1) shape

        # repeat the max-pooled features
        x = x.expand(-1, -1, x_t2.shape[2]) # expand to (batch_size, 1024, num_points)
        
        # concatenate feature transform
        x = torch.cat((x_t2, x), dim=1) # concat to (batch_size, 1088, num_points)

        # FC layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.conv4(x)

        # softmax
        x = torch.softmax(x, dim=1) # x has shape (batch_size, num_classes, num_points)

        x = x.transpose(2, 1) # (batch_size, num_points, num_classes)

        return x


class PointNetEmbedding(nn.Module):

    def __init__(self, point_dim: int = 3, sequence_len: int = 1024, embedding_dim: int = 762) -> None:
        super().__init__()

        self.point_dim = point_dim
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim

        self.feature_extractor = PointNet(point_dim)

        self.conv1 = nn.Conv1d(1024, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(256, embedding_dim * sequence_len, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # extract features
        x, x_t2 = self.feature_extractor(x)

        # max pooling
        x = torch.max(x, 2, keepdim=True)[0]

        # concatenate feature transform
        x = x.expand(-1, -1, x_t2.shape[2])
        x = torch.cat((x_t2, x), dim=1)

        # FC layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.conv4(x)

        # reshape to (batch_size, sequence_len, embedding_dim)
        x = x.view(-1, self.sequence_len, self.embedding_dim)

        return x
