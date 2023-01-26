# V2V-PoseNet-pytorch-ITOP
This is a pytorch implementation of V2V-PoseNet([V2V-PoseNet: Voxel-to-Voxel Prediction Network for Accurate 3D Hand and Human Pose Estimation from a Single Depth Map](https://arxiv.org/abs/1711.07399)) for ITOP dataset, which is largely based on the author's [torch7 implementation](https://github.com/mks0601/V2V-PoseNet_RELEASE) and pytorch reimplementation of V2V-PoseNet for MSRA hand dataset(https://github.com/dragonbook/V2V-PoseNet-pytorch).

## Dataset
Download [ITOP dataset](https://zenodo.org/record/3932973#.Y9JnvT3P1hE) and store it in **/datasets/itop/**

You won't need data for itop center.

## Start
Follow dragonbook's [installation guide](https://github.com/dragonbook/V2V-PoseNet-pytorch).

Begin with

    python experiments/main.py
  
## Visualization

    python experiments/draw_skeleton.py
  
This would produce a short video clip with estimated skeleton implemented.
