import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


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
    min_dim = 10
    max_dim = 500

    def __init__(self, trial=None):
        super().__init__()

        self.hidden_units = [400, 300, 200, 100, 50]
        self.dropout_ratios = [0.0] * 5
        if trial is not None:
            for i in range(0, 5):
                self.hidden_units[i] = trial.suggest_int(
                    f"fc{i+2}_input_dim", self.min_dim, self.max_dim
                )

        print(f"input dimentsions: {self.hidden_units}")
        layers = []
        input_dim = self.pose
        for output_dim in self.hidden_units:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim
        layers.append(nn.Linear(input_dim, self.dof))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
