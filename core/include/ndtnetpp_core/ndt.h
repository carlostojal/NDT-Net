#include <stdio.h>
#include <pthread.h>
#include <float.h>

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
void get_pointcloud_limits(float *point_cloud, int point_dim, unsigned long num_points, 
                        float *max_x, float *max_y, float *max_z,
                        float *min_x, float *min_y, float *min_z);

/*! \brief TODO: Estimate the normal distributions on the point cloud. Estimate a normal distribution per voxel of size "voxel_size".
    \param point_cloud Pointer to the point cloud.
    \param voxel_size Voxel size for distribution sampling.
    \param max_x Maximum value in the "x" dimension.
    \param max_y Maximum value in the "y" dimension.
    \param max_z Maximum value in the "z" dimension.
    \param min_x Minimum value in the "x" dimension.
    \param min_y Minimum value in the "y" dimension.
    \param min_z Minimum value in the "z" dimension.
*/
void estimate_ndt(float *point_cloud, float voxel_size, float max_x, float max_y, float max_z,
                float min_x, float min_y, float min_z);

/*! \brief TODO: Downsample the input point cloud with NDT.
    \param point_cloud Pointer to the point cloud.
    \param point_dim Point dimension. (Example: 3 for xyz points).
    \param num_points Number of points in the input point cloud.
    \param num_desired_points Number of desired points after sampling.
 */
void ndt_downsample(float *point_cloud, int point_dim, unsigned long num_points, unsigned long num_desired_points);
