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

    def __init__(self, trial):
        super().__init__()
        fc2_input_dim = trial.suggest_int("fc2_input_dim", 10, 500)
        fc3_input_dim = trial.suggest_int("fc3_input_dim", 10, 500)
        fc4_input_dim = trial.suggest_int("fc4_input_dim", 10, 500)
        fc5_input_dim = trial.suggest_int("fc5_input_dim", 10, 500)
        fc6_input_dim = trial.suggest_int("fc6_input_dim", 10, 500)
        self.fc1 = nn.Linear(self.pose, fc2_input_dim)
        self.fc2 = nn.Linear(fc2_input_dim, fc3_input_dim)
        self.fc3 = nn.Linear(fc3_input_dim, fc4_input_dim)
        self.fc4 = nn.Linear(fc4_input_dim, fc5_input_dim)
        self.fc5 = nn.Linear(fc5_input_dim, fc6_input_dim)
        self.fc6 = nn.Linear(fc6_input_dim, self.dof)

    def forward(self, x):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
            x = F.relu(layer(x))
        return self.fc6(x)
