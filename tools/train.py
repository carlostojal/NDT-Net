import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import sys
from argparse import ArgumentParser
sys.path.append(".")
from ndtnetpp.datasets.CARLA_Seg import CARLA_Seg
from ndtnetpp.models.pointnet import PointNetClassification, PointNetSegmentation

if __name__ == '__main__':

    # parse the command-line arguments
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, help="Task to perform (classification or segmentation)", default="segmentation", required=True)
    parser.add_argument("--train_path", type=str, help="Path to the training dataset", required=True)
    parser.add_argument("--val_path", type=str, help="Path to the validation dataset", required=True)
    parser.add_argument("--test_path", type=str, help="Path to the test dataset", required=True)
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=100, required=False)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=2, required=False)
    parser.add_argument("--learning_rate", type=float, help="Learning rate", default=0.001, required=False)
    parser.add_argument("--n_classes", type=int, help="Number of classes. Count with unknown/no class", default=29, required=False)
    args = parser.parse_args()

    # create the dataset
    print("Creating the dataset...", end=" ")
    if args.task == "classification":
        raise NotImplementedError("Classification task not implemented yet.")
    elif args.task == "segmentation":
        train_set = CARLA_Seg(int(args.n_classes), args.train_path)
        val_set = CARLA_Seg(int(args.n_classes), args.val_path)
        test_set = CARLA_Seg(int(args.n_classes), args.test_path)
    else:
        raise ValueError(f"Unknown task: {args.task}")
    print("done.")
    
    # create the data loaders
    print("Creating the data loaders...", end=" ")
    train_loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False)
    test_loader = DataLoader(test_set, batch_size=int(args.batch_size), shuffle=False)
    print("done.")

    # get the device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create the model
    print("Creating the model...", end=" ")
    if args.task == "classification":
        model = PointNetClassification(3, 4096, 512)
    elif args.task == "segmentation":
        model = PointNetSegmentation(3, int(args.n_classes))
    model = model.to(device)
    print("done.")

    # create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    
    # training loop
    for epoch in range(int(args.epochs)):
        print(f"Epoch {epoch}/{args.epochs}")
        # set the model to training mode
        model.train()
        for i, (pcl, gt) in enumerate(train_loader):
            # move the data to the device
            pcl = pcl.to(device)
            gt = gt.to(device)

            # forward pass
            pred = model(pcl)

            # compute the loss - cross entropy
            loss = torch.nn.functional.cross_entropy(pred, gt)

            # backward pass
            loss.backward()

            # update the weights
            optimizer.step()

            # log the loss
            print(f"\rTrain Loss: {loss.item()}", end="")
        print()

        # validation
        # set the model to evaluation mode
        model.eval()

        # disable gradient computation
        with torch.no_grad():
            for i, (pcl, gt) in enumerate(val_loader):
                # move the data to the device
                pcl = pcl.to(device)
                gt = gt.to(device)

                # forward pass
                pred = model(pcl)

                # compute the loss - cross entropy
                loss = torch.nn.functional.cross_entropy(pred, gt)

                # log the loss
                print(f"\rValidation Loss: {loss.item()}", end="")
            print()

    # test
    # set the model to evaluation mode
    model.eval()

    # disable gradient computation
    with torch.no_grad():
        for i, (pcl, gt) in enumerate(test_loader):
            # move the data to the device
            pcl = pcl.to(device)
            gt = gt.to(device)

            # forward pass
            pred = model(pcl)

            # compute the loss - cross entropy
            loss = torch.nn.functional.cross_entropy(pred, gt)

            # log the loss
            print(f"\rTest Loss: {loss.item()}", end="")
    print()

    print("Done.")
