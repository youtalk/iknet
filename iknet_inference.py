import argparse
import sys
import time

import rclpy
import torch
from open_manipulator_msgs.msg import JointPosition
from open_manipulator_msgs.srv import SetJointPosition

from iknet import IKNet


def main():
    rclpy.init(args=sys.argv)
    node = rclpy.create_node("iknet_inference")
    set_joint_position = node.create_client(SetJointPosition, "/goal_joint_space_path")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="./iknet.pth",
    )
    parser.add_argument("--trt", action="store_true", default=False)
    parser.add_argument("--x", type=float, default=0.1)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=0.1)
    parser.add_argument("--qx", type=float, default=0.0)
    parser.add_argument("--qy", type=float, default=0.0)
    parser.add_argument("--qz", type=float, default=0.0)
    parser.add_argument("--qw", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.trt:
        model = IKNet()
    else:
        from torch2trt import TRTModule
        model = TRTModule()
    model.to(device)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    i = 0
    imax = 1000
    start = time.time()
    while i < imax:
        pose = [args.x, args.y, args.z, args.qx, args.qy, args.qz, args.qw]
        if not args.trt:
            input_ = torch.FloatTensor(pose)
        else:
            input_ = torch.FloatTensor([pose])
        input_ = input_.to(device)
        # print(f"input: {input_}")
        output = model(input_)
        # print(f"output: {output}")
        i += 1
        if i % 100 == 0:
            print(i)
    elapsed = time.time() - start
    print(f"elapsed: {elapsed / imax}")
    print(f"hz: {imax / elapsed}")

    joint_position = JointPosition()
    joint_position.joint_name = [f"joint{i+1}" for i in range(4)]
    if not args.trt:
        joint_position.position = [output[i].item() for i in range(4)]
    else:
        joint_position.position = [output[0][i].item() for i in range(4)]
    request = SetJointPosition.Request()
    request.joint_position = joint_position
    request.path_time = 4.0

    future = set_joint_position.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        print(f"result: {future.result().is_planned}")
    else:
        print(f"exception: {future.exception()}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
