import argparse

import pandas as pd
import pytorch_pfn_extras as ppe
import pytorch_pfn_extras.training.extensions as extensions
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset


class IKDataset(Dataset):
    def __init__(self, kinematics_pose_csv, joint_states_csv):
        kinematics_pose = pd.read_csv(kinematics_pose_csv)
        joint_states = pd.read_csv(joint_states_csv)
        input_ = kinematics_pose.iloc[:, 3:10].values
        output = joint_states.iloc[:, 8:12].values
        self.input_ = torch.tensor(input_, dtype=torch.float32)
        self.output = torch.tensor(output, dtype=torch.float32)

    def __len__(self):
        return len(self.output)

    def __getitem__(self, index):
        return self.input_[index], self.output[index]


class IKNet(nn.Module):
    pose = 7
    dof = 4
    hidden_units = [400, 300, 200, 100, 50]

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(self.pose, self.hidden_units[0])
        self.fc2 = nn.Linear(self.hidden_units[0], self.hidden_units[1])
        self.fc3 = nn.Linear(self.hidden_units[1], self.hidden_units[2])
        self.fc4 = nn.Linear(self.hidden_units[2], self.hidden_units[3])
        self.fc5 = nn.Linear(self.hidden_units[3], self.hidden_units[4])
        self.fc6 = nn.Linear(self.hidden_units[4], self.dof)

    def forward(self, x):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
            x = F.relu(layer(x))
        return self.fc6(x)


def get_data_loaders(args):
    dataset = IKDataset(args.kinematics_pose_csv, args.joint_states_csv)
    train_size = int(len(dataset) * args.train_val_ratio)
    train_dataset = Subset(dataset, list(range(0, train_size)))
    val_dataset = Subset(dataset, list(range(train_size, len(dataset))))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader, val_loader


def train(manager, args, model, device, train_loader):
    while not manager.stop_trigger:
        model.train()
        for data, target in train_loader:
            with manager.run_iteration(step_optimizers=["main"]):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = (output - target).norm()
                ppe.reporting.report({"train/loss": loss.item() / args.batch_size})
                loss.backward()


def validate(args, model, device, data, target):
    model.eval()
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = (output - target).norm()
    ppe.reporting.report({"val/loss": loss.item() / args.batch_size})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kinematics-pose-csv", type=str, default="./kinematics_pose.csv"
    )
    parser.add_argument("--joint-states-csv", type=str, default="./joint_states.csv")
    parser.add_argument("--train-val-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--save-model", action="store_true", default=False)
    args = parser.parse_args()

    train_loader, val_loader = get_data_loaders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IKNet()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
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

    trigger = ppe.training.triggers.EarlyStoppingTrigger(
        check_trigger=(3, "epoch"), monitor="val/loss"
    )
    manager = ppe.training.ExtensionsManager(
        model,
        optimizer,
        args.epochs,
        extensions=my_extensions,
        iters_per_epoch=len(train_loader),
        stop_trigger=trigger,
    )
    train(manager, args, model, device, train_loader)

    if args.save_model:
        torch.save(model.state_dict(), "deep_learning_ik.pt")


if __name__ == "__main__":
    main()
