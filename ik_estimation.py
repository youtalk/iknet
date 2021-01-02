import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    dof = 4
    pose = 7
    hidden_units = [200, 150, 100, 50]

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(self.dof, self.hidden_units[0])
        self.fc2 = nn.Linear(self.hidden_units[0], self.hidden_units[1])
        self.fc3 = nn.Linear(self.hidden_units[1], self.hidden_units[2])
        self.fc4 = nn.Linear(self.hidden_units[2], self.hidden_units[3])
        self.fc5 = nn.Linear(self.hidden_units[3], self.pose)

    def forward(self, x):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            x = F.relu(layer(x))
        return self.fc5(x)


if __name__ == "__main__":
    net = Net()
    print(net)
