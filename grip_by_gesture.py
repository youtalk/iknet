import sys

import rclpy
from rclpy.node import Node
from open_manipulator_msgs.msg import JointPosition
from open_manipulator_msgs.srv import SetJointPosition
from std_msgs.msg import String


class GripByGesture(Node):
    delta = 0.01

    def __init__(self) -> None:
        super().__init__("grip_by_gesture")
        self.prev_time = self.get_clock().now()
        self.goal_tool_control = self.create_client(
            SetJointPosition, "/goal_tool_control")
        self.gesture_class = self.create_subscription(
            String, "/gesture_class", self.callback, 1)

    def callback(self, msg: String) -> None:
        curr_time = self.get_clock().now()
        dt = curr_time - self.prev_time
        self.prev_time = curr_time

        self.get_logger().info(msg.data)
        if msg.data in ("fist", "pan"):
            joint_position = JointPosition()
            joint_position.joint_name = ["gripper"]
            position = self.delta  # * dt.nanoseconds * 1e-9
            joint_position.position = [
                position if msg.data == "pan" else -position]
            request = SetJointPosition.Request()
            request.joint_position = joint_position
            _ = self.goal_tool_control.call_async(request)


def main():
    rclpy.init(args=sys.argv)
    node = GripByGesture()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
