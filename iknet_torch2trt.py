import argparse

import torch
from torch2trt import torch2trt

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
        default="./iknet-trt.pt",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IKNet().to(device)
    model.load_state_dict(torch.load(args.input_model))
    model.eval()
    print(model)

    input_ = torch.ones(7).to(device)
    model_trt = torch2trt(model, [input_], fp16_mode=True)
    print(model_trt)
    torch.save(model_trt.state_dict(), args.output_model)


if __name__ == "__main__":
    main()
