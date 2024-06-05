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
    parser.add_argument("--out_path", type=str, help="Path to save the model", default="out", required=False)
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=200, required=False)
    parser.add_argument("--save_every", type=int, help="Save the model every n epochs", default=10, required=False)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=16, required=False)
    parser.add_argument("--learning_rate", type=float, help="Learning rate", default=0.034, required=False)
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
    train_loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=int(args.batch_size), shuffle=False, pin_memory=True, num_workers=4)
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
    print("done.")

    LEARNING_RATE = args.learning_rate
    
    # training loop
    for epoch in range(int(args.epochs)):

        print(f"--- EPOCH {epoch+1}/{args.epochs} ---")
        # set the model to training mode
        model.train()
        curr_sample = 0
        loss_per_sample = 0
        acc = 0.0
        total_acc = 0.0
        for i, (pcl, covs, gt) in enumerate(train_loader):
            # move the data to the device
            pcl = pcl.to(device)
            covs = covs.to(device)
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
            pred = model(pcl, covs)

            # compute the loss - cross entropy
            loss = torch.nn.functional.cross_entropy(pred, gt)

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # update the weights
            optimizer.step()

            # get the loss per sample
            loss_per_sample = (loss / int(args.batch_size)).item()

            # get the accuracy (one-hot encoding)
            pred_classes = torch.argmax(pred, dim=1)
            gt_classes = torch.argmax(gt, dim=1)
            acc = torch.sum(pred_classes == gt_classes).item() / float(int(args.batch_size) * int(args.n_desired_nds))
            total_acc += acc

            # log the loss
            print(f"\rTrain Sample ({curr_sample}/{len(train_loader)*int(args.batch_size)}): train_loss: {loss_per_sample}, train_acc: {acc}, train_acc_mean: {acc_mean}", end="")

        print()

        acc_mean = acc / len(train_loader)

        # log the loss to wandb
        wandb.log({"train_loss": loss_per_sample, "train_acc": acc, "train_acc_mean": acc_mean, "epoch": epoch+1})

        # validation
        # set the model to evaluation mode
        model.eval()

        # disable gradient computation
        with torch.no_grad():
            curr_sample = 0
            acc = 0.0
            total_acc = 0.0
            for i, (pcl, covs, gt) in enumerate(val_loader):

                # move the data to the device
                pcl = pcl.to(device)
                covs = covs.to(device)
                gt = gt.to(device)

                curr_sample += int(args.batch_size)

                # forward pass
                pred = model(pcl, covs)

                # compute the loss - cross entropy
                loss = torch.nn.functional.cross_entropy(pred, gt)

                # get the loss per sample
                loss_per_sample = loss / int(args.batch_size)

                # get the accuracy (one-hot encoding)
                pred_classes = torch.argmax(pred, dim=1)
                gt_classes = torch.argmax(gt, dim=1)
                acc = torch.sum(pred_classes == gt_classes).item() / float(int(args.batch_size) * int(args.n_desired_nds))
                total_acc += acc

                # log the loss
                print(f"\rValidation Sample {curr_sample}/{len(val_loader)*int(args.batch_size)}: val_loss: {loss_per_sample.item()}, val_acc: {acc}, val_acc_mean: {acc_mean}", end="")

            print()

        acc_mean = acc / len(val_loader)
        
        # log the loss to wandb
        wandb.log({"val_loss": loss_per_sample.item(), "val_acc": acc, "val_acc_mean": acc_mean, "epoch": epoch+1})

        # save every "save_every" epochs
        if (epoch+1) % int(args.save_every) == 0:
            print("Saving the model...", end=" ")
            torch.save(model.state_dict(), f"{args.out_path}/pointnet_{epoch+1}.pth")
            print("done.")

    # test
    # set the model to evaluation mode
    model.eval()

    # disable gradient computation
    print("--- TEST ---")
    with torch.no_grad():
        curr_sample = 0
        acc = 0.0
        total_acc = 0.0
        for i, (pcl, covs, gt) in enumerate(test_loader):
            # move the data to the device
            pcl = pcl.to(device)
            covs = covs.to(device)
            gt = gt.to(device)

            curr_sample += int(args.batch_size)

            # forward pass
            pred = model(pcl, covs)

            # compute the loss - cross entropy
            loss = torch.nn.functional.cross_entropy(pred, gt)

            # get the loss per sample
            loss_per_sample = loss / int(args.batch_size)

            # get the accuracy (one-hot encoding)
            pred_classes = torch.argmax(pred, dim=1)
            gt_classes = torch.argmax(gt, dim=1)
            acc = torch.sum(pred_classes == gt_classes).item() / float(int(args.batch_size) * int(args.n_desired_nds))
            total_acc += acc

            # log the loss
            print(f"\rTest Sample {curr_sample}/{len(test_loader)*int(args.batch_size)}: test_loss: {loss_per_sample.item()}, test_acc: {acc}", end="")

    print()

    mean_acc = acc / len(test_loader)

    # log the loss to wandb
    wandb.log({"test_loss": loss_per_sample.item(), "test_acc": mean_acc})

    # finish the wandb run
    wandb.finish()

    print("Done.")
