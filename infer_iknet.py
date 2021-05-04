import argparse

import torch

from iknet import IKNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="./iknet.pt",
    )
    parser.add_argument("--x", type=float, default=0.1)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=0.1)
    parser.add_argument("--qx", type=float, default=0.0)
    parser.add_argument("--qy", type=float, default=0.0)
    parser.add_argument("--qz", type=float, default=0.0)
    parser.add_argument("--qw", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IKNet()
    model.to(device)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    input_ = torch.FloatTensor(
        [args.x, args.y, args.z, args.qx, args.qy, args.qz, args.qw]
    )
    input_ = input_.to(device)
    print(f"input: {input_}")
    output = model(input_)
    print(f"output: {output}")


if __name__ == "__main__":
    main()
