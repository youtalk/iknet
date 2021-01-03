# IKNet: Inverse kinematics neural networks for Open Manipulator X

## Data collection

## Training

```shell
python3 train_iknet.py --help
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
```

## Test

```shell
python3 test_iknet.py --help
usage: test_iknet.py [-h] [--kinematics-pose-csv KINEMATICS_POSE_CSV]
                     [--joint-states-csv JOINT_STATES_CSV] [--batch-size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --kinematics-pose-csv KINEMATICS_POSE_CSV
  --joint-states-csv JOINT_STATES_CSV
  --batch-size BATCH_SIZE
```
