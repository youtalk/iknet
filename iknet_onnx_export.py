import argparse

import torch
import torch.onnx

from iknet import IKNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-model",
        type=str,
        default="./iknet.pt",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="./iknet.onnx",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IKNet().to(device)
    model.load_state_dict(torch.load(args.input_model))
    model.eval()
    print(model)

    input_ = torch.ones(7).to(device)
    torch.onnx.export(
        model,
        input_,
        args.output_model,
        verbose=True,
        input_names=["pose"],
        output_names=["joints"],
    )


if __name__ == "__main__":
    main()
