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
        if trial is not None:
            for i in range(0, 5):
                self.hidden_units[i] = trial.suggest_int(
                    f"fc{i+2}_input_dim", self.min_dim, self.max_dim
                )

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
