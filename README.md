# IKNet: Inverse kinematics neural networks for Open Manipulator X

IKNet is an inverse kinematics estimation with simple neural networks.
This repository also contains the training and test dataset by manually moving the 4 DoF manipulator [ROBOTIS Open Manipulator X](https://emanual.robotis.com/docs/en/platform/openmanipulator_x/overview/).

IKNet can be run on [NVIDIA Jetson Nano 2GB](https://nvda.ws/2HQcb1Y), [Jetson family](https://developer.nvidia.com/EMBEDDED/Jetson-modules) or PC with/without NVIDIA GPU.

## Data collection

### Set up

Install ROS 2 on Ubuntu 18.04 by following the ROBOTIS e-Manual.

[https://emanual.robotis.com/docs/en/platform/openmanipulator_x/ros2_setup/#ros-setup](https://emanual.robotis.com/docs/en/platform/openmanipulator_x/ros2_setup/#ros-setup)

Then build some additional packages to modify a message `open_manipulator_msgs/msg/KinematicsPose`  to add timestamp.

```shell
$ mkdir -p ~/ros2/src && cd ~/ros2/src
$ git clone https://github.com/youtalk/open_manipulator.git -b kinematics-pose-header
$ git clone https://github.com/youtalk/open_manipulator_msgs.git -b kinematics-pose-header
$ cd ~/ros2
$ colcon build
$ . install/setup.bash
```

### Demo

First launch Open Manipulator X controller and turn the servo off to manually move it around.

```shell
$ ros2 launch open_manipulator_x_controller open_manipulator_x_controller.launch.py
```

```shell
$ ros2 service call /set_actuator_state open_manipulator_msgs/srv/SetActuatorState
```

Then collect the pair of the kinematics pose and the joint angles by recording `/kinematics_pose` and `/joint_states` topics under csv format.

```shell
$ ros2 topic echo --csv /kinematics_pose > kinematics_pose.csv & \
  ros2 topic echo --csv /joint_states > joint_states.csv
```

Finally append the headers into them to load by Pandas `DataFrame`.

```shell
$ sed -i "1s/^/sec,nanosec,frame_id,position_x,position_y,position_z,orientation_x,orientation_y,orientation_z,orientation_w,max_accelerations_scaling_factor,max_velocity_scaling_factor,tolerance\n/" kinematics_pose.csv
$ sed -i "1s/^/sec,nanosec,frame_id,name0,name1,name2,name3,name4,position0,position1,position2,position3,position4,velocity0,velocity1,velocity2,velocity3,velocity4,effort0,effort1,effort2,effort3,effort4\n/" joint_states.csv
```

[![IKNet data collection with Open Manipulator X](https://img.youtube.com/vi/dsHGYwkQ5Ag/0.jpg)](https://www.youtube.com/watch?v=dsHGYwkQ5Ag)

## Training

### Set up

Install PyTorch and the related packages.

```shell
$ conda install pytorch cudatoolkit=11.0 -c pytorch
$ pip3 install pytorch-pfn-extras matplotlib
```

### Demo

Train IKNet with training dataset which is inside dataset/train directory or prepared by yourself.
The training may be stopped before maximum epochs by the early stopping trigger.

```shell
$ python3 train_iknet.py --help
usage: train_iknet.py [-h] [--kinematics-pose-csv KINEMATICS_POSE_CSV]
                      [--joint-states-csv JOINT_STATES_CSV] [--train-val-ratio TRAIN_VAL_RATIO]
                      [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--lr LR] [--save-model]

optional arguments:
  -h, --help            show this help message and exit
  --kinematics-pose-csv KINEMATICS_POSE_CSV
  --joint-states-csv JOINT_STATES_CSV
  --train-val-ratio TRAIN_VAL_RATIO
  --batch-size BATCH_SIZE
  --epochs EPOCHS
  --lr LR
  --save-model

$ python3 train_iknet.py
epoch       iteration   train/loss  lr          val/loss
1           3           0.0168781   0.01        0.013123
2           6           0.0158434   0.01        0.012754
3           9           0.01492     0.01        0.0126269
...
64          192         0.00221506  0.01        0.0045823
65          195         0.00218555  0.01        0.00454468
66          198         0.00216918  0.01        0.00459203
```

## Test

### Demo

Evaluate accuracy of IKNet with test dataset which is inside dataset/test directory or prepared by yourself.

```shell
$ python3 test_iknet.py --help
usage: test_iknet.py [-h] [--kinematics-pose-csv KINEMATICS_POSE_CSV]
                     [--joint-states-csv JOINT_STATES_CSV] [--batch-size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --kinematics-pose-csv KINEMATICS_POSE_CSV
  --joint-states-csv JOINT_STATES_CSV
  --batch-size BATCH_SIZE

$ python3 test_iknet.py
Total loss = 0.006885118103027344
```
