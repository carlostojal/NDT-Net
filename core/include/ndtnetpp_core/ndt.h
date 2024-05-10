#ifndef NDT_H_
#define NDH_H_

#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

#define NUM_PCL_WORKERS 8 // number of workers for bulk point cloud processing tasks
#define PCL_DOWNSAMPLE_UPPER_THRESHOLD 0.2f // upper threshold percentage for the voxel grid downsampling

enum direction_t {
    X_POS,
    X_NEG,
    Y_POS,
    Y_NEG,
    Z_POS,
    Z_NEG,
    DIRECTION_LEN
};

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

/*! \brief Print a matrix to the standard output. 
    \param matrix Pointer to the matrix. Row-major order.
    \param rows Number of rows in the matrix.
    \param cols Number of columns in the matrix.
*/
void print_matrix(double *matrix, int rows, int cols);

/*! \brief Test matrix modification by reference. Doubles the matrix values. 
    \param matrix Pointer to the matrix. Row-major order.
    \param rows Number of rows in the matrix.
    \param cols Number of columns in the matrix.
*/
void test_modify_matrix(double *matrix, int rows, int cols);

/*! \brief Get the point cloud limits in each dimension. The values will be assigned by reference.
    \param point_cloud Pointer to the point cloud.
    \param point_dim Point dimension. (Example: 3 for xyz points).
    \param num_points Number of points in the point cloud.
    \param max_x Maximum value in the "x" dimension. Will be overwritten.
    \param max_y Maximum value in the "y" dimension. Will be overwritten.
    \param max_z Maximum value in the "z" dimension. Will be overwritten.
    \param min_x Minimum value in the "x" dimension. Will be overwritten.
    \param min_y Minimum value in the "y" dimension. Will be overwritten.
    \param min_z Minimum value in the "z" dimension. Will be overwritten.
*/
void get_pointcloud_limits(double *point_cloud, short point_dim, unsigned long num_points, 
                        double *max_x, double *max_y, double *max_z,
                        double *min_x, double *min_y, double *min_z);


/*! \brief Estimate voxel size for a number of desired points considering the limits.
    \param num_desired_voxels Number of desired voxels.
    \param max_x Maximum value in the "x" dimension.
    \param max_y Maximum value in the "y" dimension.
    \param max_z Maximum value in the "z" dimension.
    \param min_x Minimum value in the "x" dimension.
    \param min_y Minimum value in the "y" dimension.
    \param min_z Minimum value in the "z" dimension.
    \param voxel_size Estimated voxel size.
    \param len_x Number of voxels in the "x" dimension.
    \param len_y Number of voxels in the "y" dimension.
    \param len_z Number of voxels in the "z" dimension.
*/
void estimate_voxel_size(unsigned long num_desired_voxels,
                        double max_x, double max_y, double max_z,
                        double min_x, double min_y, double min_z,
                        double *voxel_size,
                        int *len_x, int *len_y, int *len_z);


/*! \brief Convert a point from metric space to voxel space (indexes).
    \param point Pointer to the point.
    \param voxel_size Voxel size.
    \param len_x Number of voxels in the "x" dimension.
    \param len_y Number of voxels in the "y" dimension.
    \param len_z Number of voxels in the "z" dimension.
    \param voxel_x Voxel index in the "x" dimension. Will be overwritten.
    \param voxel_y Voxel index in the "y" dimension. Will be overwritten.
    \param voxel_z Voxel index in the "z" dimension. Will be overwritten.
*/
int metric_to_voxel_space(double *point, double voxel_size,
                            int len_x, int len_y, int len_z,
                            unsigned int *voxel_x, unsigned int *voxel_y, unsigned int *voxel_z);


/*! \brief Convert a point from voxel space (indexes) to metric space.
    \param voxel_x Voxel index in the "x" dimension.
    \param voxel_y Voxel index in the "y" dimension.
    \param voxel_z Voxel index in the "z" dimension.
    \param len_x Number of voxels in the "x" dimension.
    \param len_y Number of voxels in the "y" dimension.
    \param len_z Number of voxels in the "z" dimension.
    \param voxel_size Voxel size.
    \param point Pointer to the point. Will be overwritten.
*/
void voxel_to_metric_space(unsigned int voxel_x, unsigned int voxel_y, unsigned int voxel_z,
                            int len_x, int len_y, int len_z,
                            double voxel_size, double *point);

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

/*! \brief Get the neighbor index in a given direction.
    \param index Index of the normal distribution in the array.
    \param len_x Number of voxels in the "x" dimension.
    \param len_y Number of voxels in the "y" dimension.
    \param len_z Number of voxels in the "z" dimension.
    \param direction Direction in the 3D space.
    \return Neighbor index.
*/
unsigned long get_neighbor_index(unsigned long index, int len_x, int len_y, int len_z, enum direction_t direction);

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
