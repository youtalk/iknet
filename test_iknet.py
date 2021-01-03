import argparse

import torch
from torch.utils.data import DataLoader

from iknet import IKNet, IKDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kinematics-pose-csv", type=str, default="./data/test/kinematics_pose.csv"
    )
    parser.add_argument(
        "--joint-states-csv", type=str, default="./data/test/joint_states.csv"
    )
    parser.add_argument("--batch-size", type=int, default=10000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IKNet()
    model.load_state_dict(torch.load("iknet.pt"))
    model.to(device)
    model.eval()

    dataset = IKDataset(args.kinematics_pose_csv, args.joint_states_csv)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    total_loss = 0.0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        total_loss += (output - target).norm().item() / args.batch_size
    print(f"Total loss = {total_loss}")


if __name__ == "__main__":
    main()
