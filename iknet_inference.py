import argparse
import sys

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

    joint_position = JointPosition()
    joint_position.joint_name = [f"joint{i+1}" for i in range(4)]
    joint_position.position = [output[i].item() for i in range(4)]
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
