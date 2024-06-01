import torch
from torch.utils.data import DataLoader
import wandb
import sys
from argparse import ArgumentParser
sys.path.append(".")
from ndtnetpp.datasets.CARLA_NDT_Seg import CARLA_NDT_Seg
from ndtnetpp.models.pointnet import PointNetClassification, PointNetSegmentation

if __name__ == '__main__':

    # parse the command-line arguments
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, help="Task to perform (classification or segmentation)", default="segmentation", required=False)
    parser.add_argument("--n_desired_nds", type=int, help="Number of desired normal distributions", default=4080, required=False)
    parser.add_argument("--train_path", type=str, help="Path to the training dataset", required=True)
    parser.add_argument("--val_path", type=str, help="Path to the validation dataset", required=True)
    parser.add_argument("--test_path", type=str, help="Path to the test dataset", required=True)
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=15, required=False)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=4, required=False)
    parser.add_argument("--learning_rate", type=float, help="Learning rate", default=0.001, required=False)
    parser.add_argument("--n_classes", type=int, help="Number of classes. Don't count with unknown/no class", default=28, required=False)
    args = parser.parse_args()

    # create the dataset
    print("Creating the dataset...", end=" ")
    if args.task == "classification":
        raise NotImplementedError("Classification task not implemented yet.")
    elif args.task == "segmentation":
        train_set = CARLA_NDT_Seg(int(args.n_classes), int(args.n_desired_nds), args.train_path)
        val_set = CARLA_NDT_Seg(int(args.n_classes), int(args.n_desired_nds), args.val_path)
        test_set = CARLA_NDT_Seg(int(args.n_classes), int(args.n_desired_nds), args.test_path)
    else:
        raise ValueError(f"Unknown task: {args.task}")
    print("done.")
    
    # create the data loaders
    print("Creating the data loaders...", end=" ")
    train_loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True, pin_memory=True)
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

    # initialize wandb
    """
    print("Initializing wandb...", end=" ")
    wandb.init(project="ndtnetpp", 
        config={
            "task": args.task,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "n_classes": args.n_classes,
            "optimizer": "Adam"
    })
    """
    print("done.")
    
    # training loop
    for epoch in range(int(args.epochs)):
        print(f"--- EPOCH {epoch+1}/{args.epochs} ---")
        # set the model to training mode
        model.train()
        curr_sample = 0
        for i, (pcl, covs, gt) in enumerate(train_loader):
            # move the data to the device
            pcl = pcl.to(device)
            covs = covs.to(device)
            gt = gt.to(device)

            curr_sample += int(args.batch_size)

            # forward pass
            pred = model(pcl, covs)

            # compute the loss - cross entropy
            loss = torch.nn.functional.cross_entropy(pred, gt)

            # backward pass
            loss.backward()

            # update the weights
            optimizer.step()

            # get the loss per sample
            loss_per_sample = loss / int(args.batch_size)

            # log the loss
            print(f"\rTrain Loss ({curr_sample}/{len(train_loader)*int(args.batch_size)}): {loss_per_sample.item()}", end="")

            del pcl, covs, gt, pred, loss_per_sample
        print()

        # log the loss to wandb
        wandb.log({"train_loss": loss.item()})

        # validation
        # set the model to evaluation mode
        model.eval()

        # disable gradient computation
        with torch.no_grad():
            for i, (pcl, covs, gt) in enumerate(val_loader):
                # move the data to the device
                pcl = pcl.to(device)
                covs = covs.to(device)
                gt = gt.to(device)

                # forward pass
                pred = model(pcl, covs)

                # compute the loss - cross entropy
                loss = torch.nn.functional.cross_entropy(pred, gt)

                # get the loss per sample
                loss_per_sample = loss / int(args.batch_size)

                # log the loss
                print(f"\rValidation Loss: {loss_per_sample.item()}", end="")

                del pcl, covs, gt, pred, loss_per_sample
            print()
        
        # log the loss to wandb
        wandb.log({"val_loss": loss_per_sample.item()})

    # test
    # set the model to evaluation mode
    model.eval()

    # disable gradient computation
    print("--- TEST ---")
    with torch.no_grad():
        for i, (pcl, covs, gt) in enumerate(test_loader):
            # move the data to the device
            pcl = pcl.to(device)
            covs = covs.to(device)
            gt = gt.to(device)

            # forward pass
            pred = model(pcl, covs)

            # compute the loss - cross entropy
            loss = torch.nn.functional.cross_entropy(pred, gt)

            # get the loss per sample
            loss_per_sample = loss / int(args.batch_size)

            # log the loss
            print(f"\rTest Loss: {loss_per_sample.item()}", end="")

            del pcl, covs, gt, pred, loss_per_sample
    print()

    # log the loss to wandb
    wandb.log({"test_loss": loss_per_sample.item()})

    # finish the wandb run
    wandb.finish()

    print("Done.")
