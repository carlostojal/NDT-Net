#ifndef NDT_H_
#define NDH_H_

/*
 MIT License

 Copyright (c) 2024 Carlos Cabaço Tojal

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

 */

#include <stdio.h>
#include <errno.h>
#include <stdbool.h>
#include <string.h>
#include <omp.h>

#include <ndtnetpp_core/normal_distributions.h>
#include <ndtnetpp_core/kullback_leibler.h>

#define DOWNSAMPLE_UPPER_THRESHOLD 0.2 // upper threshold for downsampled point cloud size
#define MIN_POINTS_GUESS 1 // minumum number of points to guess the number of normal distributions
#define MAX_POINTS_GUESS 1000000 // maximum number of points to guess the number of normal distributions
#define MIN_VOXEL_GUESS 0.05 // minimum voxel size guess
#define MAX_VOXEL_GUESS 5.0 // maximum voxel size guess
#define MAX_GUESS_ITERATIONS 10 // maximum number of iterations to guess the number of normal distributions

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Prune normal distributions with small divergence until the desired number is reached.
    \param nd_array Pointer to the array of normal distributions.
    \param len_x Number of voxels in the "x" dimension.
    \param len_y Number of voxels in the "y" dimension.
    \param len_z Number of voxels in the "z" dimension.
    \param num_desired_nds Number of desired normal distributions.
    \param num_valid_nds Pointer to the number of valid normal distributions. Will be overwritten.
    \param kl_divergences Pointer to the array of Kullback-Leibler divergences. Will be overwritten.
    \param num_kl_divergences Pointer to the number of Kullback-Leibler divergences. Will be overwritten.
*/
int prune_nds(struct normal_distribution_t *nd_array, 
                    unsigned int len_x, unsigned int len_y, unsigned int len_z,
                    unsigned long num_desired_nds, unsigned long *num_valid_nds,
                    struct kl_divergence_t *kl_divergences, unsigned long *num_kl_divergences);


/*! \brief Get a point cloud, covariances and classes from an array of normal distributions. 
    \param nd_array Pointer to the array of normal distributions.
    \param len_x Number of voxels in the "x" dimension.
    \param len_y Number of voxels in the "y" dimension.
    \param len_z Number of voxels in the "z" dimension.
    \param voxel_size Voxel size.
    \param point_cloud Pointer to the point cloud. Will be overwritten.
    \param num_points Pointer to the number of points in the point cloud. Will be overwritten.
    \param covariances Pointer to the array of covariances. Will be overwritten.
    \param classes Pointer to the array of classes. Will be overwritten.
*/
int to_point_cloud(struct normal_distribution_t *nd_array, 
                    unsigned int len_x, unsigned int len_y, unsigned int len_z,
                    double offset_x, double offset_y, double offset_z,
                    double voxel_size,
                    double *point_cloud, unsigned long *num_points,
                    double *covariances,
                    unsigned short *classes);

/*! \brief Downsample the input point cloud with NDT.
    \param point_cloud Pointer to the point cloud.
    \param point_dim Point dimension. (Example: 3 for xyz points).
    \param num_points Number of points in the input point cloud.
    \param classes Point classes array.
    \param num_classes Number of classes.
    \param num_desired_points Number of desired points after sampling.
    \param downsampled_point_cloud Pointer to the downsampled point cloud. Will be overwritten.
    \param num_downsampled_points Number of points in the downsampled point cloud. Will be overwritten.
    \param covariances Pointer to the array of covariances. Will be overwritten.
    \param downsampled_classes Pointer to the downsampled point classes. Will be overwritten.
 */
int ndt_downsample(double *point_cloud, unsigned short point_dim, unsigned long num_points, 
                    unsigned short *classes, unsigned short num_classes,
                    unsigned long num_desired_points,
                    double *downsampled_point_cloud, unsigned long *num_downsampled_points,
                    double *covariances,
                    unsigned short *downsampled_classes,
                    struct normal_distribution_t *nd_array,
                    struct kl_divergence_t *kl_divergences, unsigned long *num_kl_divergences);

#ifdef __cplusplus
}
#endif

#endif // NDT_H_
