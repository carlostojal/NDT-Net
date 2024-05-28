import torch
import torch.nn as nn
import sys
sys.path.append(".")

from ndtnetpp.models.pointnet import PointNetClassification

if __name__ == '__main__':
    # get the device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a tensor
    x = torch.randn(2, 100000, 3).uniform_(-50.0, 50.0)
    print(x)
    x = x.to(device)

    # create the model
    model = PointNetClassification(3, 4096, 512)
    model = model.to(device)
    
    # forward pass
    out = model(x)
    
    # print the output
    # print(out.shape)
