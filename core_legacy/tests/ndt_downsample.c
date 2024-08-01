#include <stdio.h>
#include <stdlib.h>
#include <ndnet_core/ndt.h>

#define NUM_POINTS 90000
#define POINT_DIM 3

#define NUM_DOWNSAMPLED_POINTS 24

#define NUM_CLASSES 28

#define NUM_LOOPS 10

int main(int argc, char *argv[]) {

    (void)argc;
    (void)argv;

    srand(0);

    for(unsigned int i = 0; i < NUM_LOOPS; i++) {

        // allocate a point cloud
        double pointcloud[NUM_POINTS * POINT_DIM] = {0};
        // fill with random values
        for(unsigned long i = 0; i < NUM_POINTS * POINT_DIM; i++) {
            pointcloud[i] = (double)rand() / RAND_MAX;
        }

        // allocate a downsampled point cloud
        double downsampled[NUM_DOWNSAMPLED_POINTS * POINT_DIM];

        // allocate the covariances
        double covariances[NUM_DOWNSAMPLED_POINTS * 9];

        // downsample the point cloud
        unsigned long num_points;
        unsigned long num_downsampled_points;
        unsigned int len_x, len_y, len_z;
        double offset_x, offset_y, offset_z;
        double voxel_size;
        struct normal_distribution_t *nd_array = NULL;
        unsigned long num_valid_nds;
        struct kl_divergence_t *kl_divergences;
        unsigned long num_kl_divergences;
        
        if(ndt_downsample(pointcloud, POINT_DIM, NUM_POINTS,
                        &len_x, &len_y, &len_z,
                        &offset_x, &offset_y, &offset_z,
                        &voxel_size,
                        NULL, 0,
                        NUM_DOWNSAMPLED_POINTS,
                        downsampled, &num_downsampled_points,
                        covariances,
                        NULL,
                        &nd_array, &num_valid_nds,
                        &kl_divergences, &num_kl_divergences) < 0) {
            fprintf(stderr, "Error downsampling the point cloud!\n");
            return -1;
        }

        // free the normal distributions
        free_nds(nd_array, len_x * len_y * len_z);
        free_kl_divergences(kl_divergences);
    }

    return 0;
}