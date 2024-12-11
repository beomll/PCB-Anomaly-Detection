# Anomaly_Detection

## Requirements
```
pytorch 2.3.0+cu12 => pytorch.org
wandb 0.16.6 => pip install wandb
```

## Quick Start
```
python run.py [-h] [--seed SEED] [-m MODEL_TYPE] [-lr LEARNING_RATE] [-e NUM_EPOCHS] [-b BATCH_SIZE] [-sch SCHEDULER] [-opt OPTIMIZER]\
              [-w WARMUP_STEPS] [-g GAMMA] [-d DROP_PROB] [-s STEP_SIZE]\
              [-trn TRAIN_DATA] [-val VAL_DATA] [-tst TEST_DATA] [-p PRETRAINED] [-mp MODEL_PATH] [-sp SAVE_PATH]
```
```
python run.py -lr 1e-4 -e 250 -b 256 -opt adam -g 0.1 -p False -d 0.1 -m resnext
```
