# deep-learning-ik
Inverse kinematics estimation of Open Manipulator X with neural networks

## Training

```shell
python3 train_ik.py --help
usage: train_ik.py [-h] [--kinematics-pose-csv KINEMATICS_POSE_CSV]
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
