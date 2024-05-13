#ifndef NDT_H_
#define NDH_H_

#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

#define NUM_PCL_WORKERS 8 // number of workers for bulk point cloud processing tasks
#define PCL_DOWNSAMPLE_UPPER_THRESHOLD 0.2f // upper threshold percentage for the voxel grid downsampling

struct normal_distribution_t {
    double mean[3]; // xyz mean of the distribution (3-d)
    double old_mean[3]; // last mean iteration (3-d)
    double covariance[3]; // flattened covariance matrix (9-d)
    double m2[3]; // sum of squared differences. used to compute variances
    long num_samples; // number of samples
    bool being_updated; // flag to indicate if the distribution is being updated
};

struct pcl_worker_args_t {
    double *point_cloud; // pointer to the point cloud
    unsigned long num_points; // number of points in the point cloud
    struct normal_distribution_t *nd_array; // pointer to the array of normal distributions
    pthread_mutex_t *mutex_array; // pointer to the array of mutexes
    pthread_cond_t *cond_array; // pointer to the array of condition variables
    double voxel_size; // voxel size for distribution sampling
    int len_x; // number of voxels in the "x" dimension
    int len_y; // number of voxels in the "y" dimension
    int len_z; // number of voxels in the "z" dimension
    int worker_id; // worker id
};

struct dk_divergence_t {
    double divergence; // divergence value
    struct normal_distribution_t *p; // pointer to the first normal distribution
    struct normal_distribution_t *q; // pointer to the second normal distribution
};

/*! \brief Worker routine for normal distribution update. */
void *pcl_worker(void *arg);

/*! \brief Estimate the normal distributions on the point cloud. Estimate a normal distribution per voxel of size "voxel_size".
    \param point_cloud Pointer to the point cloud.
    \param voxel_size Voxel size for distribution sampling.
    \param len_x Number of voxels in the "x" dimension.
    \param len_y Number of voxels in the "y" dimension.
    \param len_z Number of voxels in the "z" dimension.
    \param nd_array Pointer to the array of normal distributions. Will be overwritten.
    \param num_nds Number of normal distributions. Will be overwritten.
*/
int estimate_ndt(double *point_cloud, unsigned long num_points, double voxel_size,
                    int len_x, int len_y, int len_z,
                    struct normal_distribution_t *nd_array);

/*! \brief Compute the multivariate Kullback-Leibler divergence between two normal distributions.
    \param p Pointer to the first normal distribution.
    \param q Pointer to the second normal distribution.
    \param divergence Pointer to the divergence value. Will be overwritten.
*/
void dk_divergence(struct normal_distribution_t *p, struct normal_distribution_t *q, double *divergence);

/*! \brief Collapse normal distributions with small divergence until the desired number is reached.
    \param nd_array Pointer to the array of normal distributions.
    \param len_x Number of voxels in the "x" dimension.
    \param len_y Number of voxels in the "y" dimension.
    \param len_z Number of voxels in the "z" dimension.
    \param num_desired_nds Number of desired normal distributions.
*/
void collapse_nds(struct normal_distribution_t *nd_array, int len_x, int len_y, int nel_z,
                    unsigned long num_desired_nds, unsigned long *num_valid_nds);

/*! \brief Downsample the input point cloud with NDT.
    \param point_cloud Pointer to the point cloud.
    \param point_dim Point dimension. (Example: 3 for xyz points).
    \param num_points Number of points in the input point cloud.
    \param num_desired_points Number of desired points after sampling.
    \param downsampled_point_cloud Pointer to the downsampled point cloud. Will be overwritten.
    \param num_downsampled_points Number of points in the downsampled point cloud. Will be overwritten.
 */
void ndt_downsample(double *point_cloud, short point_dim, unsigned long num_points, unsigned long num_desired_points,
                    double *downsampled_point_cloud, unsigned long *num_downsampled_points);

#endif // NDT_H_
