#include <stdio.h>
#include <pthread.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>

#define NUM_PCL_WORKERS 8 // number of workers for bulk point cloud processing tasks
#define PCL_DOWNSAMPLE_UPPER_THRESHOLD 0.2f // upper threshold percentage for the voxel grid downsampling

struct normal_distribution_t {
    float *mean; // xyz mean of the distribution (3-d)
    float *covariance; // flattened covariance matrix (9-d)
    long num_samples; // number of samples
};

/*! \brief Print a matrix to the standard output. 
    \param matrix Pointer to the matrix. Row-major order.
    \param rows Number of rows in the matrix.
    \param cols Number of columns in the matrix.
*/
void print_matrix(float *matrix, int rows, int cols);

/*! \brief Test matrix modification by reference. Doubles the matrix values. 
    \param matrix Pointer to the matrix. Row-major order.
    \param rows Number of rows in the matrix.
    \param cols Number of columns in the matrix.
*/
void test_modify_matrix(float *matrix, int rows, int cols);

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
void get_pointcloud_limits(float *point_cloud, short point_dim, unsigned long num_points, 
                        float *max_x, float *max_y, float *max_z,
                        float *min_x, float *min_y, float *min_z);


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
                        float max_x, float max_y, float max_z,
                        float min_x, float min_y, float min_z,
                        float *voxel_size,
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
int metric_to_voxel_space(float *point, float voxel_size,
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
                            float voxel_size, float *point);

/*! \brief TODO: Estimate the normal distributions on the point cloud. Estimate a normal distribution per voxel of size "voxel_size".
    \param point_cloud Pointer to the point cloud.
    \param voxel_size Voxel size for distribution sampling.
    \param len_x Number of voxels in the "x" dimension.
    \param len_y Number of voxels in the "y" dimension.
    \param len_z Number of voxels in the "z" dimension.
*/
int estimate_ndt(float *point_cloud, float voxel_size,
                    int len_x, int len_y, int len_z);

/*! \brief TODO: Downsample the input point cloud with NDT.
    \param point_cloud Pointer to the point cloud.
    \param point_dim Point dimension. (Example: 3 for xyz points).
    \param num_points Number of points in the input point cloud.
    \param num_desired_points Number of desired points after sampling.
 */
void ndt_downsample(float *point_cloud, short point_dim, unsigned long num_points, unsigned long num_desired_points);
