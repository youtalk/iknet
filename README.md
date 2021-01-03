# deep-learning-ik
Inverse kinematics estimation of Open Manipulator X with neural networks

## Training

```shell
python3 ik_estimation.py --help
usage: ik_estimation.py [-h] [--kinematics-pose-csv KINEMATICS_POSE_CSV]
                        [--joint-states-csv JOINT_STATES_CSV] [--train-test-ratio TRAIN_TEST_RATIO]
                        [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--lr LR] [--save-model]

optional arguments:
  -h, --help            show this help message and exit
  --kinematics-pose-csv KINEMATICS_POSE_CSV
  --joint-states-csv JOINT_STATES_CSV
  --train-test-ratio TRAIN_TEST_RATIO
  --batch-size BATCH_SIZE
  --epochs EPOCHS
  --lr LR
  --save-model
```

## Demo
