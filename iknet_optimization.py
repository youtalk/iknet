import argparse

import optuna
import pytorch_pfn_extras as ppe
import pytorch_pfn_extras.training.extensions as extensions
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from iknet import IKDataset, IKNet

args = None


def get_data_loaders(args):
    dataset = IKDataset(args.kinematics_pose_csv, args.joint_states_csv)
    train_size = int(len(dataset) * args.train_val_ratio)
    train_dataset = Subset(dataset, list(range(0, train_size)))
    val_dataset = Subset(dataset, list(range(train_size, len(dataset))))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader, val_loader


def train(manager, args, model, device, train_loader):
    result = float("inf")
    while not manager.stop_trigger:
        model.train()
        for data, target in train_loader:
            with manager.run_iteration(step_optimizers=["main"]):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = (output - target).norm()
                result = loss.item() / args.batch_size
                ppe.reporting.report({"train/loss": result})
                loss.backward()

    return result


def validate(args, model, device, data, target):
    model.eval()
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = (output - target).norm()
    ppe.reporting.report({"val/loss": loss.item() / args.batch_size})


def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IKNet(trial)
    model.to(device)
    train_loader, val_loader = get_data_loaders(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    trigger = ppe.training.triggers.EarlyStoppingTrigger(
        check_trigger=(3, "epoch"), monitor="val/loss"
    )
    my_extensions = [
        extensions.LogReport(),
        extensions.ProgressBar(),
        extensions.observe_lr(optimizer=optimizer),
        extensions.ParameterStatistics(model, prefix="model"),
        extensions.VariableStatisticsPlot(model),
        extensions.Evaluator(
            val_loader,
            model,
            eval_func=lambda data, target: validate(args, model, device, data, target),
            progress_bar=True,
        ),
        extensions.PlotReport(["train/loss", "val/loss"], "epoch", filename="loss.png"),
        extensions.PrintReport(
            [
                "epoch",
                "iteration",
                "train/loss",
                "lr",
                "val/loss",
            ]
        ),
    ]
    manager = ppe.training.ExtensionsManager(
        model,
        optimizer,
        args.epochs,
        extensions=my_extensions,
        iters_per_epoch=len(train_loader),
        stop_trigger=trigger,
    )
    return train(manager, args, model, device, train_loader)


def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    trial = study.best_trial

    print("value: ", trial.value)
    print("params: ")
    for key, value in trial.params.items():
        print("  {}: {}".format(key, value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kinematics-pose-csv",
        type=str,
        default="./dataset/train/kinematics_pose.csv",
    )
    parser.add_argument(
        "--joint-states-csv", type=str, default="./dataset/train/joint_states.csv"
    )
    parser.add_argument("--train-val-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    main()
