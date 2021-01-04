# IKNet: Inverse kinematics neural networks for ROBOTIS Open Manipulator X

IKNet is an inverse kinematics estimation with simple neural networks.
This repository also contains the training and test dataset by manually moving the 4 DoF manipulator [ROBOTIS Open Manipulator X](https://emanual.robotis.com/docs/en/platform/openmanipulator_x/overview/).

IKNet can be trained on tested on [NVIDIA Jetson Nano 2GB](https://nvda.ws/2HQcb1Y), [Jetson family](https://developer.nvidia.com/EMBEDDED/Jetson-modules) or PC with/without NVIDIA GPU.
The training needs 900MB of GPU memory under default options.

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

Train IKNet with training dataset which is inside dataset/train directory or prepared by yourself. The dataset/train dataset contains a 5-minutes movement at 100 [Hz] sampling.

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
1           3           0.0188889   0.01        0.0130676
2           6           0.0165503   0.01        0.0132546
3           9           0.0167138   0.01        0.0134633
...
61          183         0.00267084  0.01        0.00428417
62          186         0.00266047  0.01        0.00461381
63          189         0.00260262  0.01        0.00461737
```

The training can be run on NVIDIA Jetson Nano 2GB.

[![IKNet training on NVIDIA Jetson Nano 2GB](https://img.youtube.com/vi/R_RtWAhCt8o/0.jpg)](https://www.youtube.com/watch?v=R_RtWAhCt8o)

The loss indicates the L1 norm of the joint angles. So the final networks solved 0.00461737 [rad] accuracy on average.

![train/loss and val/loss](https://user-images.githubusercontent.com/579333/103491840-44a38880-4e6a-11eb-946c-222c46b97878.png)

## Test

### Demo

Evaluate accuracy of IKNet with test dataset which is inside dataset/test directory or prepared by yourself.
The dataset/test dataset contains a 1-minute movement at 100 [Hz] sampling.

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

## Reference

- Theofanidis, Michail & Sayed, Saif & Cloud, Joe & Brady, James & Makedon, Fillia. (2018). Kinematic Estimation with Neural Networks for Robotic Manipulators: 27th International Conference on Artificial Neural Networks, Rhodes, Greece, October 4–7, 2018, Proceedings, Part III. 10.1007/978-3-030-01424-7_77. 
- Duka, Adrian-Vasile. (2014). Neural Network based Inverse Kinematics Solution for Trajectory Tracking of a Robotic Arm. Procedia Technology. 12. 20–27. 10.1016/j.protcy.2013.12.451.
