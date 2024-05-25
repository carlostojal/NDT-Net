#include <ndtnetpp_core/ndt.h>

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

void prune_nds(struct normal_distribution_t *nd_array, int len_x, int len_y, int len_z,
                    unsigned long num_desired_nds, unsigned long *num_valid_nds) {

    // compare the divergences in neighboring voxels
    // the distributions with the lowest divergence will be removed, because they introduce the least new information

    if(num_desired_nds > len_x * len_y * len_z) {
        fprintf(stderr, "Number of desired normal distributions is greater than the number of voxels!\n");
        return;
    }

    *num_valid_nds = 0;

    // keep an ordered array of divergences
    unsigned long divergences_len = 0;
    struct kl_divergence_t *divergences = (struct kl_divergence_t *) malloc(len_x * len_y * len_z * DIRECTION_LEN * sizeof(struct kl_divergence_t));
    if(divergences == NULL) {
        fprintf(stderr, "Error allocating memory for divergences: %s\n", strerror(errno));
        return;
    }

    // debug the normal distributions
    // print_nds(nd_array, len_x, len_y, len_z);

    // calculate the divergences between each pair of neighboring distributions
    #pragma omp parallel for
    for(int z = 0; z < len_z; z++) {
        for(int y = 0; y < len_y; y++) {
            for(int x = 0; x < len_x; x++) {

                // get the index of the current voxel
                unsigned long index;
                if(voxel_pos_to_index(x, y, z, len_x, len_y, len_z, &index) < 0) {
                    fprintf(stderr, "Error converting voxel position to index!\n");
                    return;
                }

                // verify if the voxel has samples
                if(nd_array[index].num_samples == 0)
                    continue;
                (*num_valid_nds)++;

                // calculate the divergence between the current voxel and the neighbors in each direction
                for(short i = 0; i < DIRECTION_LEN; i++) {

                    // get the neighbor index
                    unsigned long neighbor_index;
                    if(get_neighbor_index(index, len_x, len_y, len_z, i, &neighbor_index) == -4) { // neighbor out of bounds
                        continue;
                    } else if (neighbor_index < 0) {
                        fprintf(stderr, "Error getting neighbor index!\n");
                        return;
                    }

                    // verify if the other voxel has samples
                    if(nd_array[neighbor_index].num_samples == 0)
                        continue;
                    
                    // calculate the divergence between the distributions
                    double div = 0;
                    if(kl_divergence(&nd_array[index], &nd_array[neighbor_index], &div) == -2) {
                        // the q covariance matrix is singular
                        continue;
                    }

                    // insert the divergence in the ordered array
                    unsigned long j = 0;
                    while(j < divergences_len) {
                        if(divergences[j].divergence < div)
                            break;
                        j++;
                    }
                    // shift the divergences to the right
                    for(unsigned long k = divergences_len; k > j; k--) {
                        divergences[k] = divergences[k-1];
                    }
                    // insert the divergence
                    divergences[j].divergence = div;
                    divergences[j].p = &nd_array[index];
                    divergences[j].q = &nd_array[neighbor_index];
                    divergences_len++;
                }
            }
        }
    }

    // remove the distributions with the smallest divergence until the desired number is reached
    unsigned int to_remove = *num_valid_nds - num_desired_nds;
    unsigned long idx_to_remove = 0;

    // remove the distributions with the smallest divergence
    if(*num_valid_nds > num_desired_nds) {
        for(unsigned long i = 0; i < to_remove; idx_to_remove++) {
            // if it was already removed or has no samples, pick another
            if(divergences[idx_to_remove].p->num_samples == 0) {
                continue;
            }
            // set the number of samples to 0
            divergences[idx_to_remove].p->num_samples = 0;
            (*num_valid_nds)--;
            divergences_len--;
            i++;
        }
    }

    // free the divergences array
    free(divergences);
}

int ndt_downsample(double *point_cloud, unsigned short point_dim, unsigned long num_points,
                    unsigned short *classes, unsigned short num_classes,
                    unsigned long num_desired_points,
                    double *downsampled_point_cloud, unsigned long *num_downsampled_points,
                    double *covariances,
                    unsigned short *downsampled_classes) {

    // get the point cloud limits
    double max_x, max_y, max_z;
    double min_x, min_y, min_z;
    get_pointcloud_limits(point_cloud, point_dim, num_points, &max_x, &max_y, &max_z, &min_x, &min_y, &min_z);

    // estimate the voxel size
    double voxel_size;
    int len_x, len_y, len_z;
    double x_offset, y_offset, z_offset;

    // create a grid of normal distributions
    struct normal_distribution_t *nd_array = NULL;

    double guess = (double) (MAX_VOXEL_GUESS - MIN_VOXEL_GUESS) / 2.0;
    double min_guess = MIN_VOXEL_GUESS;
    double max_guess = MAX_VOXEL_GUESS;

    unsigned long num_nds;
    unsigned int iter = 0;
    do {

        estimate_voxel_grid(max_x, max_y, max_z, min_x, min_y, min_z, guess, &len_x, &len_y, &len_z,
                            &x_offset, &y_offset, &z_offset);

        // allocate the normal distributions
        nd_array = (struct normal_distribution_t *) realloc(nd_array, len_x * len_y * len_z * sizeof(struct normal_distribution_t));
        if(nd_array == NULL) {
            fprintf(stderr, "Error allocating memory for normal distributions: %s\n", strerror(errno));
            return -1;
        }

        if(estimate_ndt(point_cloud, num_points, classes, num_classes, guess, len_x, len_y, len_z, x_offset, y_offset, z_offset, nd_array, &num_nds) < 0) {
            fprintf(stderr, "Error estimating normal distributions!\n");
            return -2;
        }

        // adjust the voxel size guess limits for binary search
        if(num_nds > num_desired_points * (1+DOWNSAMPLE_UPPER_THRESHOLD)) {
            min_guess = guess;
        } else if(num_nds < num_desired_points) {
            max_guess = guess;
        } else {
            // reached a valid number of normal distributions
            break;
        }

        // get the next guess
        guess = min_guess + (max_guess - min_guess) / 2.0;

        iter++;

    } while(iter < MAX_GUESS_ITERATIONS);

    voxel_size = guess;

    if(iter == MAX_GUESS_ITERATIONS) {
        fprintf(stderr, "Reached maximum number of iterations!\n");
        return -3;
    }

    // compute the divergences and remove the distributions with the smallest divergence
    unsigned long num_valid_nds;
    prune_nds(nd_array, len_x, len_y, len_z, num_desired_points, &num_valid_nds);

    // downsample the point cloud, iterating the voxels
    unsigned long downsampled_index = 0;
    #pragma omp parallel for
    for(int z = 0; z < len_z; z++) {
        for(int y = 0; y < len_y; y++) {
            for(int x = 0; x < len_x; x++) {

                // get the index of the current voxel
                unsigned long index;
                if(voxel_pos_to_index(x, y, z, len_x, len_y, len_z, &index) < 0) {
                    fprintf(stderr, "Error converting voxel position to index!\n");
                    return -4;
                }

                // verify if the voxel has samples
                if(nd_array[index].num_samples == 0)
                    continue;

                // get the point in metric space
                double point[3];
                voxel_to_metric_space(x, y, z, len_x, len_y, len_z, x_offset, y_offset, z_offset, voxel_size, point);

                // copy the point to the downsampled point cloud
                for(int i = 0; i < 3; i++) {
                    downsampled_point_cloud[downsampled_index*3 + i] = point[i];
                }
                // copy the covariance matrix
                memcpy(&covariances[downsampled_index*9], nd_array[index].covariance, 9 * sizeof(double));
                // copy the class
                if(classes != NULL) {
                    downsampled_classes[downsampled_index] = nd_array[index].class;
                }

                downsampled_index++;
            }
        }
    }

    // free the normal distributions
    // free the classes count pointer
    for(unsigned long i = 0; i < len_x * len_y * len_z; i++) {
        if(nd_array[i].num_class_samples != NULL)
            free(nd_array[i].num_class_samples);
    }
    free(nd_array);

    // set the number of downsampled points
    *num_downsampled_points = downsampled_index;

    // print_matrix(downsampled_point_cloud, *num_downsampled_points, 3);

    return 0;
}
