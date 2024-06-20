import torch
from torch.utils.data import DataLoader
import optuna
from typing import List
import sys
from argparse import ArgumentParser
sys.path.append(".")
from ndtnetpp.datasets.CARLA_NDT_Seg import CARLA_Seg
from ndtnetpp.models.ndtnet import NDTNetSegmentation

# parse the command-line arguments
parser = ArgumentParser()
parser.add_argument("--num_trials", type=int, help="Number of trials", default=100, required=False)
parser.add_argument("--num_points", type=int, help="Number of points", default=4080, required=False)
parser.add_argument("--num_classes", type=int, help="Number of classes", default=29, required=False)
parser.add_argument("--dataset_path", type=str, help="Path to the dataset", required=True)
args = parser.parse_args()

NUM_POINTS = int(args.num_points)
NUM_CLASSES = int(args.num_classes)
DATASET_PATH = str(args.dataset_path)

def objective(trial: optuna.trial.Trial) -> float:

    # optimizers list to choose from
    optimizers: List = [torch.optim.Adam, torch.optim.SGD]

    # suggest hyperparameters
    opt = trial.suggest_int("optimizer", 0, 1)
    bs = trial.suggest_int("batch_size", 2, 10)
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1)

    # create the dataset a data loader
    train_set = CARLA_Seg(NUM_CLASSES, NUM_POINTS, DATASET_PATH)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=4)

    # get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create the model
    model = NDTNetSegmentation(3, NUM_CLASSES)    
    model = model.to(device)

    # create the optimizer
    optim = optimizers[opt](model.parameters(), lr=lr)

    # set the model to training mode
    model.train()

    for epoch in range(10):

        print(f"--- EPOCH {epoch+1} ---")

        print(f"Opt: {optimizers[opt]}, BS: {bs}, LR: {lr}")
        
        curr_sample = 0
        for i, (pcl, covs, gt) in enumerate(train_loader):
            # move the data to the device
            pcl = pcl.to(device)
            covs = covs.to(device)
            gt = gt.to(device)

            # skip batch if it has only one sample
            if pcl.shape[0] == 1:
                continue

            curr_sample += bs

            # forward pass
            pred = model(pcl, covs)

            # compute the loss
            loss = torch.nn.functional.cross_entropy(pred, gt)

            # backward pass
            optim.zero_grad()
            loss.backward()

            # update the weights
            optim.step()

            print(f"\rLoss {curr_sample}/{len(train_loader)*bs}: {loss.item()}", end="")

        return loss

if __name__ == '__main__':

    # create the study
    study = optuna.create_study(study_name="ndtnet_seg", direction="minimize")
    study.optimize(objective, n_trials=int(args.num_trials))

    # get the best trial
    best_trial = study.best_trial
    print(f"Best trial: {best_trial.number}")
    print(f"\tValue: {best_trial.value}")
    print("\tParams: ")
    for key, value in best_trial.params.items():
        print(f"\t- {key}: {value}")
