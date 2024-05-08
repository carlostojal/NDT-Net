#include <ndtnetpp_core/ndt.h>

void print_matrix(float *matrix, int rows, int cols) {
    // Print matrix
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            printf("%f ", matrix[i*cols + j]);
        }
        printf("\n");
    }
}

void test_modify_matrix(float *matrix, int rows, int cols) {
    
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            matrix[i*cols + j] = matrix[i*cols + j] * 2;
        }
    }
}

void get_pointcloud_limits(float *point_cloud, int point_dim, unsigned long num_points,
                        float *max_x, float *max_y, float *max_z,
                        float *min_x, float *min_y, float *min_z) {

    *max_x = FLT_MIN;
    *max_y = FLT_MIN;
    *max_z = FLT_MIN;
    *min_x = FLT_MIN;
    *min_x = FLT_MIN;
    *min_y = FLT_MIN;
    *min_z = FLT_MIN;

    // iterate over the points
    for(unsigned long i = 0; i < num_points; i++) {

        // verify the maximum x
        if(point_cloud[i*point_dim] > *max_x)
            *max_x = point_cloud[i*point_dim];
        // verify the minimum x
        if(point_cloud[i*point_dim] < *min_x)
            *min_x = point_cloud[i*point_dim];

        // verify the maximum y
        if(point_cloud[i*point_dim + 1] > *max_y)
            *max_y = point_cloud[i*point_dim + 1];
        // verify the minimum y
        if(point_cloud[i*point_dim + 1] < *min_y)
            *min_y = point_cloud[i*point_dim + 1];

        // verify the maximum z
        if(point_cloud[i*point_dim + 2] > *max_z)
            *max_z = point_cloud[i*point_dim + 2];
        // verify the minimum z
        if(point_cloud[i*point_dim + 2] < *min_z)
            *min_z = point_cloud[i*point_dim + 2];
        
    }
}
