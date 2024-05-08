# NDT-Net++
PointNet++-based point cloud processing neural network using NDT-based sampling and grouping.

PointNet++ has several sampling and grouping stages during its inference. Traditionally, they use farthest point sampling (FPS) and neighborhood grouping.

In this work, as in the standard NDT, a normal distribution is computed for each voxel, getting to a point cloud dimension as close as possible to the desired point count. To remove the extra distributions over the desired point count, the Kullback-Leibler divergence is computed between neighborhing distributions. The distributions with the least divergence, which theoretically are the most redundant, are combined.

The rest of the PointNet and PointNet++ architecture is as defined by the original paper, with the exception that points are 12D vectors instead of 3D. Each point sample is represented by the concatenation of the normal distribution mean (3D) and its flattened covariance matrix (9D).

## Setup

### Requirements
- Python
- PyTorch
- Make and GCC

### Instructions
TODO
