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

int prune_nds(struct normal_distribution_t *nd_array, 
                    unsigned int len_x, unsigned int len_y, unsigned int len_z,
                    unsigned long num_desired_nds, unsigned long *num_valid_nds,
                    struct kl_divergence_t *kl_divergences, unsigned long *num_kl_divergences) {

    // compare the divergences in neighboring voxels
    // the distributions with the lowest divergence will be removed, because they introduce the least new information

    if(num_desired_nds > *num_valid_nds) {
        fprintf(stderr, "Number of desired normal distributions is greater than the number valid distributions!\n");
        return -1;
    }

    // debug the normal distributions
    // print_nds(nd_array, len_x, len_y, len_z);    

    // remove the distributions with the smallest divergence until the desired number is reached
    unsigned int to_remove = *num_valid_nds - num_desired_nds;
    unsigned long idx_to_remove = 0;

    // remove the distributions with the smallest divergence
    for(unsigned long i = 0; i < to_remove; idx_to_remove++) {

        // printf("div: %p\n", kl_divergences);

        if(idx_to_remove >= *num_kl_divergences) {
            fprintf(stderr, "Reached the end of the divergences array!\n");
            return -2;
        }

        // if it was already removed or has no samples, skip this
        if(kl_divergences[idx_to_remove].p->num_samples == 0) {
            continue;
        }
        // set the number of samples to 0, invalidating the normal distribution
        kl_divergences[idx_to_remove].p->num_samples = 0;
        (*num_valid_nds)--;
        (*num_kl_divergences)--;
        i++;
    }

    // move the divergences array idx_to_remove positions to the left
    for(unsigned long i = 0; i < *num_kl_divergences; i++) {
        kl_divergences[i] = kl_divergences[i+idx_to_remove];
    }
}

int to_point_cloud(struct normal_distribution_t *nd_array,
                    unsigned int len_x, unsigned int len_y, unsigned int len_z,
                    double x_offset, double y_offset, double z_offset,
                    double voxel_size,
                    double *point_cloud, unsigned long *num_points,
                    double *covariances,
                    unsigned short *classes) {


    *num_points = 0;

    // downsample the point cloud, iterating the voxels
    #pragma omp parallel for
    for(int z = 0; z < len_z; z++) {
        for(int y = 0; y < len_y; y++) {
            for(int x = 0; x < len_x; x++) {

                // get the index of the current voxel
                unsigned long index;
                if(voxel_pos_to_index(x, y, z, len_x, len_y, len_z, &index) < 0) {
                    fprintf(stderr, "Error converting voxel position to index!\n");
                    return -1;
                }

                // verify if the voxel has samples
                if(nd_array[index].num_samples == 0)
                    continue;

                // get the point in metric space
                double point[3];
                voxel_to_metric_space(x, y, z, len_x, len_y, len_z, x_offset, y_offset, z_offset, voxel_size, point);

                // copy the point to the downsampled point cloud
                memcpy(&point_cloud[(*num_points)*3], nd_array[index].mean, 3 * sizeof(double));

                // copy the covariance matrix
                memcpy(&covariances[(*num_points)*9], nd_array[index].covariance, 9 * sizeof(double));
                // copy the class
                if(classes != NULL) {
                    classes[*num_points] = nd_array[index].class;
                }

                (*num_points)++;
            }
        }
    }
}

int ndt_downsample(double *point_cloud, unsigned short point_dim, unsigned long num_points,
                    unsigned int *len_x, unsigned int *len_y, unsigned int *len_z,
                    double *offset_x, double *offset_y, double *offset_z,
                    double *voxel_size,
                    unsigned short *classes, unsigned short num_classes,
                    unsigned long num_desired_points,
                    double *downsampled_point_cloud, unsigned long *num_downsampled_points,
                    double *covariances,
                    unsigned short *downsampled_classes,
                    struct normal_distribution_t **nd_array, unsigned long *num_valid_nds,
                    struct kl_divergence_t **kl_divergences, unsigned long *num_kl_divergences) {

    // get the point cloud limits
    double max_x, max_y, max_z;
    double min_x, min_y, min_z;
    get_pointcloud_limits(point_cloud, point_dim, num_points, &max_x, &max_y, &max_z, &min_x, &min_y, &min_z);

    double guess = (double) (MAX_VOXEL_GUESS - MIN_VOXEL_GUESS) / 2.0;
    double min_guess = MIN_VOXEL_GUESS;
    double max_guess = MAX_VOXEL_GUESS;

    unsigned long num_nds;
    unsigned int iter = 0;
    do {

        // estimate the voxel grid size, dimensions and offsets
        estimate_voxel_grid(max_x, max_y, max_z, min_x, min_y, min_z, guess, len_x, len_y, len_z,
                            offset_x, offset_y, offset_z);

        // allocate the normal distributions
        *nd_array = (struct normal_distribution_t *) malloc((*len_x) * (*len_y) * (*len_z) * sizeof(struct normal_distribution_t));
        if(*nd_array == NULL) {
            fprintf(stderr, "Error allocating memory for normal distributions: %s\n", strerror(errno));
            return -1;
        }

        // estimate the normal distributions, voxelizing the point cloud
        if(estimate_ndt(point_cloud, num_points, 
                        classes, num_classes, 
                        guess, 
                        *len_x, *len_y, *len_z, 
                        *offset_x, *offset_y, *offset_z, 
                        *nd_array, &num_nds) < 0) {
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

        // free the normal distributions
        free_nds(*nd_array, (*len_x) * (*len_y) * (*len_z));

        iter++;

    } while(iter < MAX_GUESS_ITERATIONS);

    *voxel_size = guess;

    if(iter == MAX_GUESS_ITERATIONS) {
        fprintf(stderr, "Reached maximum number of iterations!\n");
        return -3;
    }

    // compute the divergences
    // allocate the divergences array
    *kl_divergences = (struct kl_divergence_t *) malloc((*len_x) * (*len_y) * (*len_z) * DIRECTION_LEN * sizeof(struct kl_divergence_t));
    if(*kl_divergences == NULL) {
        fprintf(stderr, "Error allocating memory for divergences: %s\n", strerror(errno));
        return -4;
    }
    if(calculate_kl_divergences(*nd_array, *len_x, *len_y, *len_z, num_valid_nds, *kl_divergences, num_kl_divergences) < 0) {
        fprintf(stderr, "Error calculating divergences!\n");
        return -5;
    }
    
    // remove the distributions with the smallest divergence
    prune_nds(*nd_array, *len_x, *len_y, *len_z, num_desired_points, num_valid_nds, *kl_divergences, num_kl_divergences);

    // convert to point cloud
    to_point_cloud(*nd_array, *len_x, *len_y, *len_z, 
                    *offset_x, *offset_y, *offset_z, 
                    *voxel_size, 
                    downsampled_point_cloud, num_downsampled_points, 
                    covariances, 
                    downsampled_classes);

    // print_matrix(downsampled_point_cloud, *num_downsampled_points, 3);

    return 0;
}

void free_nds(struct normal_distribution_t *nd_array, unsigned long num_nds) {

    // iterate the normal distributions to free the class samples array
    for(unsigned long i = 0; i < num_nds; i++) {
        if(nd_array[i].num_samples > 0) {
            free(nd_array[i].num_class_samples);
        }
    }

    // free the normal distributions array
    free(nd_array);

    // assign the pointer to NULL for clarity
    nd_array = NULL;
}
