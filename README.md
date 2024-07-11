# NDT-Net
PointNet-based point cloud processing neural network using NDT-based sampling and grouping.

Traditionally, farthest point sampling (FPS) is used to reduce point cloud dimensionality. However, that approach leads to some degree of loss of information, becoming more noticeable with harsher downsamples.

In this work, as in the standard NDT, a normal distribution is computed for each voxel, getting to normal distribution count as close as possible to the desired normal distribution point count. To remove the extra distributions over the desired point count, the Kullback-Leibler divergence is computed between neighborhing distributions in all directions. The distributions with the least divergence, which theoretically are well modeled by a neighbor, are the most redundant and are hence removed.

The rest of the PointNet architecture is as defined by the original paper, with the exception that points are 12D vectors instead of 3D. Each point sample is represented by the concatenation of the normal distribution mean (3D) and its flattened covariance matrix (9D).

## Setup

### Dependencies
- Python 3
- PyTorch
- CMake, Make and GCC
- GNU Scientific Library (GSL) (libgsl)
- Open3D

### Instructions

#### Bare metal
- Install the dependencies.
- In the `core` subdirectory, create a `build` subdirectory and navigate there.
- From that `build` subdirectory:
    - Issue the command `cmake ..`;
    - Issue the command `make`.

#### Docker
- Run the command ```docker build -t ndtnet .```.

In case you are using Visual Studio Code, a Dev Container configuration is available as well as debug configurations, allowing a simple setup and running.

## Training

You can update and use the Visual Studio Code debug configuration created, or instead run the command:
```python tools/train.py --epochs 130 --batch_size 16 --n_desired_nds 1000 --train_path /path/to/dataset/train --val_path /path/to/dataset/validation --test_path /path/to/dataset/test --out_path /path/to/out/ndtnet```
