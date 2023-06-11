# OKDPH

This repository contains the code for CVPR2023 OKDPH： [Generalization Matters: Loss Minima Flattening via Parameter Hybridization for Efficient Online Knowledge Distillation](https://arxiv.org/abs/2303.14666).



## Data

3 datasets were used in the paper:

* CIFAR-10
* CIFAR-100
* ImageNet: Downloadable from https://image-net.org/download.php

For downloaded data sets please place them in the 'dataset' folder.

dataset:

-- cifar-10-batches-py

-- cifar-100-python

## Requirements

* PyTorch 1.0 or higher
* Python 3.6



## Run
```bash
cd src
bash OKDPH.sh
```

For the case of four students: 

```bash
cd src
python OKDPH.py --omega 0.8 --beta 0.8 --gamma 0.5 --interval 1_epoch \
    --model_names resnet32 resnet32 resnet32 resnet32 \
    --transes hflip cutout augment auto_aug base \
    --log 21_cifar10_okdph_4stu_1ep
```

Please refer to the bash files for more running commands.


## Baselines
```bash
cd src
bash baseline.sh
```


## Experiment

* [Loss Landscape](experiment/landscape/resnet32/draw.ipynb)
* [Sample Limited Data](experiment/sample)

