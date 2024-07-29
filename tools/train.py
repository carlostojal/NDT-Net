import torch
from torch.utils.data import DataLoader
import numpy as np
import wandb
import sys
import os
import datetime
from typing import Tuple
from argparse import ArgumentParser
sys.path.append(".")
from ndtnetpp.datasets.CARLA_Seg import CARLA_Seg
from ndtnetpp.models.ndtnet import NDTNetClassification, NDTNetSegmentation
from ndtnetpp.preprocessing.ndtnet_preprocessing import ndt_preprocessing


def run_one_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loader: DataLoader, device: torch.device, args: ArgumentParser, epoch: int, mode: str="train")-> Tuple[float, float, float, float]:
    """
    Run one epoch of the training/validation/test loop.

    Args:
        model (torch.nn.Module): model to be trained/evaluated
        optimizer (torch.optim.Optimizer): optimizer to be used
        loader (DataLoader): data loader
        device (torch.device): device to run the computations
        args (ArgumentParser): command-line arguments
        epoch (int): current epoch
        mode (str, optional): mode of the loop (train, val, test). Defaults to "train".

    Returns:
        Tuple[float, float, float, float]: loss, mean loss, accuracy, mean accuracy
    """

    # set the model to training mode
    if mode == "train":
        model.train()
    else:
        model.eval()

    curr_sample = 0
    loss = 0.0
    acc = 0.0
    total_acc = 0.0
    total_loss = 0.0
    for i, (pcl, gt) in enumerate(loader):
        # move the data to the device
        pcl = pcl.to(device) # [B, R, N, 3], where R is the number of resolutions
        gt = gt.to(device)

        # skip batch if it has only one sample
        if pcl.shape[0] == 1:
            continue

        # adjust learning rate
        if epoch+1 % 20 == 0:
            LEARNING_RATE *= 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE

        curr_sample += int(args.batch_size)

        # forward pass
        # remove the "1" dimension
        pcl = pcl.squeeze(1)
        gt = gt.squeeze(1) # (batch_size, num_points, num_classes)

        # preprocess the batch
        pcl, covs, gt = ndt_preprocessing(int(args.n_desired_nds), pcl, gt, int(args.n_classes))

        pred = model(pcl, covs) # (batch_size, num_nds, num_classes)

        # compute the loss - cross entropy
        loss = torch.nn.functional.cross_entropy(pred, gt)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        loss = loss.item()
        total_loss += loss.item()

        # update the weights
        optimizer.step()

        # get the accuracy (one-hot encoding)
        pred_classes = torch.argmax(pred, dim=2)
        gt_classes = torch.argmax(gt, dim=2)
        acc = torch.sum(pred_classes == gt_classes).item() / float(int(args.batch_size) * int(args.n_desired_nds))
        total_acc += acc

        # log the loss
        print(f"{mode} sample ({curr_sample}/{len(loader)*int(args.batch_size)}): {mode}_loss: {loss}, {mode}_acc: {acc}", end=" ")

    return loss, total_loss / len(loader), acc, total_acc / len(loader)


if __name__ == '__main__':

    # parse the command-line arguments
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, help="Task to perform (classification or segmentation)", default="segmentation", required=False)
    parser.add_argument("--n_desired_nds", type=int, help="Number of desired normal distributions", default=2080, required=False)
    parser.add_argument("--n_samples", type=int, help="Number of samples to take initially using FPS", default=70000, required=False)
    parser.add_argument("--train_path", type=str, help="Path to the training dataset", required=True)
    parser.add_argument("--val_path", type=str, help="Path to the validation dataset", required=True)
    parser.add_argument("--test_path", type=str, help="Path to the test dataset", required=True)
    parser.add_argument("--out_path", type=str, help="Path to save the model", default="out", required=False)
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=200, required=False)
    parser.add_argument("--save_every", type=int, help="Save the model every n epochs", default=2, required=False)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=16, required=False)
    parser.add_argument("--learning_rate", type=float, help="Learning rate", default=0.034, required=False)
    parser.add_argument("--n_classes", type=int, help="Number of classes. Don't count with unknown/no class", default=28, required=False)
    parser.add_argument("--feature_dim", type=int, help="Dimension of the feature vectors.", default=768, required=False)
    args = parser.parse_args()

    path = os.path.join(args.out_path, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # get the device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.set_default_device(device)

    # create the dataset
    print("Creating the dataset...", end=" ")
    if "classification" in args.task:
        raise NotImplementedError("Classification task not implemented yet.")
    elif "segmentation" in args.task:
        train_set = CARLA_Seg(int(args.n_classes), int(args.n_samples), args.train_path)
        val_set = CARLA_Seg(int(args.n_classes), int(args.n_samples), args.val_path)
        test_set = CARLA_Seg(int(args.n_classes), int(args.n_samples), args.test_path)
    else:
        raise ValueError(f"Unknown task: {args.task}")
    print("done.")
    
    # create the data loaders
    print("Creating the data loaders...", end=" ")
    # generator = torch.Generator(device=device)
    train_loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=int(args.batch_size), shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=int(args.batch_size), shuffle=True, pin_memory=True, num_workers=4)
    print("done.")

    # create the model
    print("Creating the model...", end=" ")
    if args.task == "classification":
        model = NDTNetClassification(point_dim=3, num_classes=512)
    elif args.task == "segmentation":
        model = NDTNetSegmentation(point_dim=3, num_classes=int(args.n_classes), feature_dim=int(args.feature_dim))
    model = model.to(device)
    print("done.")

    # create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))

    # initialize wandb
    print("Initializing wandb...", end=" ")
    """
    wandb.init(project="ndtnetpp",
        name=f"{args.task}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "task": args.task,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "n_classes": args.n_classes,
            "n_distributions": args.n_desired_nds,
            "n_samples": args.n_samples,
            "optimizer": "Adam"
    })
    """
    print("done.")

    LEARNING_RATE = args.learning_rate
    
    # training loop
    for epoch in range(int(args.epochs)):

        print(f"--- EPOCH {epoch+1}/{args.epochs} ---")

        # train
        loss, mean_loss, acc, mean_acc = run_one_epoch(model, optimizer, train_loader, device, args, epoch, mode="train")
        # wandb.log({"train_loss": loss, "train_loss_mean": mean_loss, "train_acc": acc, "train_acc_mean": mean_acc, "epoch": epoch+1})
        print()

        # validation
        loss, mean_loss, acc, mean_acc = run_one_epoch(model, optimizer, val_loader, device, args, epoch, mode="val")
        # wandb.log({"val_loss": loss, "val_loss_mean": mean_loss, "val_acc": acc, "val_acc_mean": mean_acc, "epoch": epoch+1})
        print()

        # save every "save_every" epochs
        if (epoch+1) % int(args.save_every) == 0:
            # create the output path if it doesn't exist
            if not os.path.exists(path):
                os.makedirs(path)
            print("Saving the model...", end=" ")
            torch.save(model.state_dict(), f"{path}/ndtnet_{args.task}_full_{epoch+1}.pth")
            # save the feature extractor
            torch.save(model.feature_extractor.state_dict(), f"{path}/ndtnet_{args.task}_backbone_{epoch+1}.pth")
            print("done.")

        del loss, mean_loss, acc, mean_acc

    # test
    loss, mean_loss, acc, mean_acc = run_one_epoch(model, optimizer, test_loader, device, args, epoch, mode="test")
    # wandb.log({"test_loss": loss, "test_loss_mean": mean_loss, "test_acc": acc, "test_acc_mean": mean_acc})
    print()

    del loss, mean_loss, acc, mean_acc

    # finish the wandb run
    wandb.finish()

    print("Done.")
