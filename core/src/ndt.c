#include <ndtnetpp_core/ndt.h>

void print_matrix(double *matrix, int rows, int cols) {
    // Print matrix
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            printf("%f ", matrix[i*cols + j]);
        }
        printf("\n");
    }
}

void test_modify_matrix(double *matrix, int rows, int cols) {
    
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            matrix[i*cols + j] = matrix[i*cols + j] * 2;
        }
    }
}

void get_pointcloud_limits(double *point_cloud, short point_dim, unsigned long num_points,
                        double *max_x, double *max_y, double *max_z,
                        double *min_x, double *min_y, double *min_z) {

    *max_x = DBL_MIN;
    *max_y = DBL_MIN;
    *max_z = DBL_MIN;
    *min_x = DBL_MAX;
    *min_x = DBL_MAX;
    *min_y = DBL_MAX;
    *min_z = DBL_MAX;

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

void estimate_voxel_size(unsigned long num_desired_voxels,
                        double max_x, double max_y, double max_z,
                        double min_x, double min_y, double min_z,
                        double *voxel_size,
                        int *len_x, int *len_y, int *len_z) {


    // calculate the lengths in each dimension
    double x_dim = max_x - min_x;
    double y_dim = max_y - min_y;
    double z_dim = max_z - min_z;

    // calculate the voxel size
    *voxel_size = (double) num_desired_voxels / x_dim;
    *voxel_size /= y_dim;
    *voxel_size /= z_dim;

    *voxel_size = floor(*voxel_size);
}

int metric_to_voxel_space(double *point, double voxel_size,
                            int len_x, int len_y, int len_z,
                            unsigned int *voxel_x, unsigned int *voxel_y, unsigned int *voxel_z) {

    // the center voxel of the grid is at metric (0, 0, 0)

    // find the origin of the grid in metric space
    double x_origin, y_origin, z_origin;
    x_origin = -((double) len_x / 2) * voxel_size;
    y_origin = -((double) len_y / 2) * voxel_size;
    z_origin = -((double) len_z / 2) * voxel_size;

    // calculate the voxel indexes
    *voxel_x = (unsigned int) floor((point[0] - x_origin) / voxel_size);
    *voxel_y = (unsigned int) floor((point[1] - y_origin) / voxel_size);
    *voxel_z = (unsigned int) floor((point[2] - z_origin) / voxel_size);

    // check if the point is outside the grid
    if(*voxel_x < 0 || *voxel_x >= len_x ||
        *voxel_y < 0 || *voxel_y >= len_y ||
        *voxel_z < 0 || *voxel_z >= len_z) {
        fprintf(stderr, "Point outside the grid!\n");
        return -1;
    }

    return 0;
}

void voxel_to_metric_space(unsigned int voxel_x, unsigned int voxel_y, unsigned int voxel_z,
                            int len_x, int len_y, int len_z,
                            double voxel_size, double *point) {

    // the center voxel of the grid is at metric (0, 0, 0)

    // find the origin of the grid in metric space
    double x_origin, y_origin, z_origin;
    x_origin = -((double) len_x / 2) * voxel_size;
    y_origin = -((double) len_y / 2) * voxel_size;
    z_origin = -((double) len_z / 2) * voxel_size;

    // calculate the point in metric space
    point[0] = x_origin + voxel_x * voxel_size;
    point[1] = y_origin + voxel_y * voxel_size;
    point[2] = z_origin + voxel_z * voxel_size;
}

void *pcl_worker(void *arg) {

    // get the worker arguments
    struct pcl_worker_args_t *args = (struct pcl_worker_args_t *) arg;

    // get the point range for the worker from the worker id
    unsigned long start = args->worker_id * (args->num_points/ NUM_PCL_WORKERS);
    unsigned long end = (args->worker_id + 1) * (args->num_points / NUM_PCL_WORKERS);

    // iterate over the points
    for(unsigned long i = start; i < end; i++) {

        // check if the point cloud is finished
        if(i <= args->num_points)
            break;

        // get the voxel indexes for the point
        unsigned int voxel_x, voxel_y, voxel_z;
        if(metric_to_voxel_space(&args->point_cloud[i*3], args->voxel_size, args->len_x, args->len_y, args->len_z, &voxel_x, &voxel_y, &voxel_z) < 0) {
            fprintf(stderr, "Error converting point to voxel space!\n");
            return NULL;
        }
        unsigned long voxel_index = voxel_x * args->len_y * args->len_z + voxel_y * args->len_z + voxel_z;

        // lock the mutex for the voxel
        if(pthread_mutex_lock(&args->mutex_array[voxel_index]) != 0) {
            fprintf(stderr, "Error locking distribution mutex: %s\n", strerror(errno));
            return NULL;
        }

        // wait for the condition variable
        while(args->nd_array[voxel_index].being_updated) {
            if(pthread_cond_wait(&args->cond_array[voxel_index], &args->mutex_array[voxel_index]) != 0) {
                fprintf(stderr, "Error waiting for condition variable: %s\n", strerror(errno));
                return NULL;
            }
        }

        // update the normal distribution for the voxel
        args->nd_array[voxel_index].num_samples++;
        for(int j = 0; j < 3; j++) {
            // copy the old mean
            args->nd_array[voxel_index].old_mean[j] = args->nd_array[voxel_index].mean[j];
            // update the mean
            args->nd_array[voxel_index].mean[j] += (args->point_cloud[i*3+j] - args->nd_array[voxel_index].mean[j]) / args->nd_array[voxel_index].num_samples;
            // update the sum of squared differences for the variances
            args->nd_array[voxel_index].m2[j] += (args->point_cloud[i*3+j] - args->nd_array[voxel_index].old_mean[j]) * (args->point_cloud[i*3+j] - args->nd_array[voxel_index].mean[j]);

            // iterate the other dimensions to update the covariance matrix
            for(int k = 0; k < 3; k++) {
                // it's a diagonal, so it's a variance and not a covariance
                if(j == k)
                    continue;

                // update the covariance matrix
                args->nd_array[voxel_index].covariance[j*3+k] += (args->point_cloud[i*3+j] - args->nd_array[voxel_index].mean[j]) * (args->point_cloud[i*3+k] - args->nd_array[voxel_index].mean[k]) / args->nd_array[voxel_index].num_samples;            
            }
        }
        
        // unlock the mutex for the voxel
        if(pthread_mutex_unlock(&args->mutex_array[voxel_index]) != 0) {
            fprintf(stderr, "Error unlocking distribution mutex: %s\n", strerror(errno));
            return NULL;
        }

        // signal the condition variable
        if(pthread_cond_signal(&args->cond_array[voxel_index]) != 0) {
            fprintf(stderr, "Error signaling condition variable: %s\n", strerror(errno));
            return NULL;
        }
    }
}


int estimate_ndt(double *point_cloud, unsigned long num_points, double voxel_size,
                    int len_x, int len_y, int len_z,
                    struct normal_distribution_t *nd_array) {


    // create an array of normal distributions, one per voxel
    nd_array = (struct normal_distribution_t *) malloc(len_x * len_y * len_z * sizeof(struct normal_distribution_t));
    if(nd_array == NULL) {
        fprintf(stderr, "Error allocating memory for normal distributions: %s\n", strerror(errno));
        return -1;
    }
    for(int i = 0; i < len_x * len_y * len_z; i++) {
        // initialize the normal distributions
        nd_array[i].num_samples = 0;
        for(int j = 0; j < 3; j++) {
            nd_array[i].mean[j] = 0;
            nd_array[i].m2[j] = 0;
        }
        nd_array[i].being_updated = false;
    }

    // create an array of mutexes, one per voxel
    pthread_mutex_t *mutex_array = (pthread_mutex_t *) malloc(len_x * len_y * len_z * sizeof(pthread_mutex_t));
    if(mutex_array == NULL) {
        fprintf(stderr, "Error allocating memory for distribution mutexes: %s\n", strerror(errno));
        return -2;
    }

    // initialize the mutexes
    for(int i = 0; i < len_x * len_y * len_z; i++) {
        if(pthread_mutex_init(&mutex_array[i], NULL) != 0) {
            fprintf(stderr, "Error initializing distribution mutex: %s\n", strerror(errno));
            return -3;
        }
    }

    // create an array of condition variables, one per voxel
    pthread_cond_t *cond_array = (pthread_cond_t *) malloc(len_x * len_y * len_z * sizeof(pthread_cond_t));
    if(cond_array == NULL) {
        fprintf(stderr, "Error allocating memory for condition variables: %s\n", strerror(errno));
        return -5;
    }

    // initialize the condition variables
    for(int i = 0; i < len_x * len_y * len_z; i++) {
        if(pthread_cond_init(&cond_array[i], NULL) != 0) {
            fprintf(stderr, "Error initializing condition variable: %s\n", strerror(errno));
            return -6;
        }
    }

    // allocate a pool of threads
    pthread_t *threads = (pthread_t *) malloc(NUM_PCL_WORKERS * sizeof(pthread_t));
    if(threads == NULL) {
        fprintf(stderr, "Error allocating memory for threads: %s\n", strerror(errno));
        return -7;
    }

    // create an array of worker arguments
    struct pcl_worker_args_t *args_array = (struct pcl_worker_args_t *) malloc(NUM_PCL_WORKERS * sizeof(struct pcl_worker_args_t));
    if(args_array == NULL) {
        fprintf(stderr, "Error allocating memory for worker arguments: %s\n", strerror(errno));
        return -8;
    }

    // create the threads
    for(int i = 0; i < NUM_PCL_WORKERS; i++) {

        // create the worker arguments
        struct pcl_worker_args_t *args = &args_array[i];
        args->point_cloud = point_cloud;
        args->num_points = num_points;
        args->nd_array = nd_array;
        args->mutex_array = mutex_array;
        args->cond_array = cond_array;
        args->voxel_size = voxel_size;
        args->len_x = len_x;
        args->len_y = len_y;
        args->len_z = len_z;
        args->worker_id = i;

        if(pthread_create(&threads[i], NULL, pcl_worker, (void *) args) != 0) {
            fprintf(stderr, "Error creating thread: %s\n", strerror(errno));
            return -9;
        }
    }

    // wait for the threads to finish
    for(int i = 0; i < NUM_PCL_WORKERS; i++) {
        if(pthread_join(threads[i], NULL) != 0) {
            fprintf(stderr, "Error joining thread: %s\n", strerror(errno));
            return -10;
        }
    }

    // destroy the mutexes
    // destroy distribution mutexes
    for(int i = 0; i < len_x * len_y * len_z; i++) {
        if(pthread_mutex_destroy(&mutex_array[i]) < 0) {
            fprintf(stderr, "Error destroying distribution mutex: %s\n", strerror(errno));
            return -7;
        }
    }

    // free the array of mutexes
    free(mutex_array);

    // free the array of condition variables
    free(cond_array);

    // free the pool of threads
    free(threads);

    // free the array of worker arguments
    free(args_array);

    // return 0 in case of success
    return 0;  
}

void dk_divergence(struct normal_distribution_t *p, struct normal_distribution_t *q, double *divergence) {

    // calculate the divergence between two normal distributions
    // the divergence is the multivariate Kullback-Leibler divergence

    // create GSL matrices from the covariance matrices
    gsl_matrix_view p_covariance = gsl_matrix_view_array(p->covariance, 3, 3);
    gsl_matrix_view q_covariance = gsl_matrix_view_array(q->covariance, 3, 3);

    // make the LU decomposition of the covariance matrices
    gsl_matrix *p_LU = gsl_matrix_alloc(3, 3);
    gsl_matrix *q_LU = gsl_matrix_alloc(3, 3);
    gsl_permutation *p_permutation = gsl_permutation_alloc(3);
    gsl_permutation *q_permutation = gsl_permutation_alloc(3);
    int p_signum, q_signum;
    gsl_linalg_LU_decomp(&p_covariance.matrix, p_permutation, &p_signum);
    gsl_linalg_LU_decomp(&q_covariance.matrix, q_permutation, &q_signum);
    gsl_permutation_free(p_permutation);

    // calculate the determinant of the covariance matrices
    double p_det = gsl_linalg_LU_det(&p_covariance.matrix, p_signum);
    double q_det = gsl_linalg_LU_det(&q_covariance.matrix, q_signum);

    // calculate the difference between the means
    gsl_matrix *mean_diff = gsl_matrix_alloc(3, 1); // allocate the mean difference vector
    gsl_matrix_view p_mean = gsl_matrix_view_array(p->mean, 3, 1);
    gsl_matrix_view q_mean = gsl_matrix_view_array(q->mean, 3, 1);
    gsl_matrix_memcpy(mean_diff, &p_mean.matrix); // copy the p mean to the difference
    gsl_matrix_sub(mean_diff, &q_mean.matrix); // subtract the q mean from the difference
    // transpose the mean difference vector in a copy
    gsl_matrix *mean_diff_transpose = gsl_matrix_alloc(1, 3);
    gsl_matrix_transpose_memcpy(mean_diff_transpose, mean_diff);

    // calculate the inverse of the q covariance matrix
    gsl_matrix *q_inverse = gsl_matrix_alloc(3, 3);
    gsl_linalg_LU_invert(&q_covariance.matrix, q_permutation, q_inverse);
    gsl_permutation_free(q_permutation);

    // calculate the trace of the multiplication of the inverse of the q covariance matrix and the p covariance matrix
    gsl_matrix *trace_matrix = gsl_matrix_alloc(3, 3);
    gsl_matrix_memcpy(trace_matrix, q_inverse); // copy the q inverse to the trace matrix
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, q_inverse, &p_covariance.matrix, 0.0, trace_matrix); // multiply the q inverse by the p covariance matrix
    double trace = 0;
    for(int i = 0; i < 3; i++) {
        trace += gsl_matrix_get(trace_matrix, i, i);
    }

    // fist part of the divergence (mean difference transposed * q inverse * mean difference)
    gsl_matrix *first_part = gsl_matrix_alloc(1, 3);
    gsl_matrix_memcpy(first_part, mean_diff_transpose); // copy the mean difference transpose to the first part
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, first_part, q_inverse, 0.0, first_part); // multiply the first part by the q inverse
    double *first_part_result = 0;
    // convert the first part and mean difference to a GSL vector
    gsl_vector_view first_part_view = gsl_vector_view_array(first_part->data, 3);
    gsl_vector_view mean_diff_view = gsl_vector_view_array(mean_diff->data, 3);
    gsl_blas_ddot(&first_part_view.vector, &mean_diff_view.vector, first_part_result); // calculate the dot product of the first part and the mean difference

    // calculate the divergence
    *divergence = 0.5 * (*first_part_result + trace - log(q_det/p_det) - 3);

    // free the allocated memory
    gsl_matrix_free(p_LU);
    gsl_matrix_free(q_LU);
    gsl_matrix_free(mean_diff);
    gsl_matrix_free(mean_diff_transpose);
    gsl_matrix_free(q_inverse);
    gsl_matrix_free(trace_matrix);
    gsl_matrix_free(first_part);
}

unsigned long get_neighbor_index(unsigned long index, int len_x, int len_y, int len_z, enum direction_t direction) {

    if(index < 0 || index >= len_x * len_y * len_z) {
        fprintf(stderr, "Invalid index for neighbor divergence!\n");
        return -1;
    }

    // get the indexes of the neighbor
    short direction_x = 0;
    short direction_y = 0;
    short direction_z = 0;
    switch(direction) {
        case X_POS:
            direction_x = 1;
            break;
        case Y_POS:
            direction_y = 1;
            break;
        case Z_POS:
            direction_z = 1;
            break;
        case X_NEG:
            direction_x = -1;
            break;
        case Y_NEG:
            direction_y = -1;
            break;
        case Z_NEG:
            direction_z = -1;
            break;
        default:
            fprintf(stderr, "Invalid direction for neighbor divergence!\n");
            return -1;
    }
    
    return index + direction_x * len_y * len_z + direction_y * len_z + direction_z;
}

void collapse_nds(struct normal_distribution_t *nd_array, int len_x, int len_y, int len_z,
                    unsigned long num_desired_nds, unsigned long *num_valid_nds) {

    // compare the divergences in neighboring voxels
    // the distributions with the lowest divergence will be removed, because they introduce the least new information

    *num_valid_nds = 0;

    // keep an ordered array of divergences
    unsigned long divergences_len = 0;
    struct dk_divergence_t *divergences = (struct dk_divergence_t *) malloc(len_x * len_y * len_z * DIRECTION_LEN * sizeof(struct dk_divergence_t));
    if(divergences == NULL) {
        fprintf(stderr, "Error allocating memory for divergences: %s\n", strerror(errno));
        return;
    }

    // TODO: parallelize this loop
    // calculate the divergences between each pair of neighboring distributions
    for(int x = 1; x < len_x - 1; x++) {
        for(int y = 1; y < len_y - 1; y++) {
            for(int z = 1; z < len_z - 1; z++) {

                // get the index of the current voxel
                unsigned long index = x * len_y * len_z + y * len_z + z;

                // verify if the voxel has samples
                if(nd_array[index].num_samples == 0)
                    continue;
                *num_valid_nds++;

                // calculate the divergence between the current voxel and the neighbors in each direction
                for(short i = 0; i < DIRECTION_LEN; i++) {

                    // get the neighbor index
                    unsigned long neighbor_index = get_neighbor_index(index, len_x, len_y, len_z, i);
                    if(neighbor_index < 0) {
                        fprintf(stderr, "Error getting neighbor index!\n");
                        return;
                    }

                    // verify if the other voxel has samples
                    if(nd_array[neighbor_index].num_samples == 0)
                        continue;
                    
                    // calculate the divergence between the distributions
                    double div = 0;
                    dk_divergence(&nd_array[index], &nd_array[neighbor_index], &div);

                    // insert the divergence in the ordered array
                    unsigned long j = 0;
                    while(j < divergences_len && divergences[j].divergence < div) {
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

    // remove the distributions with the smallest divergence
    for(unsigned long i = 0; i < *num_valid_nds - num_desired_nds; i++) {
        // set the number of samples to 0
        divergences[i].p->num_samples = 0;
        *num_valid_nds--;
    }

    // free the divergences array
    free(divergences);
}

void ndt_downsample(double *point_cloud, short point_dim, unsigned long num_points, unsigned long num_desired_points,
                    double *downsampled_point_cloud, unsigned long *num_downsampled_points) {

    // get the point cloud limits
    double max_x, max_y, max_z;
    double min_x, min_y, min_z;
    get_pointcloud_limits(point_cloud, point_dim, num_points, &max_x, &max_y, &max_z, &min_x, &min_y, &min_z);

    // estimate the voxel size
    double voxel_size;
    int len_x, len_y, len_z;
    estimate_voxel_size(num_desired_points, max_x, max_y, max_z, min_x, min_y, min_z, &voxel_size, &len_x, &len_y, &len_z);

    // create a grid of normal distributions
    struct normal_distribution_t *nd_array;
    if(estimate_ndt(point_cloud, num_points, voxel_size, len_x, len_y, len_z, nd_array) < 0) {
        fprintf(stderr, "Error estimating normal distributions!\n");
        return;
    }

    // compute the divergences and remove the distributions with the smallest divergence
    unsigned long num_valid_nds;
    collapse_nds(nd_array, len_x, len_y, len_z, num_desired_points, &num_valid_nds);

    // TODO: parallelize this loop
    // downsample the point cloud
    unsigned long downsampled_index = 0;
    for(int x = 0; x < len_x; x++) {
        for(int y = 0; y < len_y; y++) {
            for(int z = 0; z < len_z; z++) {

                // get the index of the current voxel
                unsigned long index = x * len_y * len_z + y * len_z + z;

                // verify if the voxel has samples
                if(nd_array[index].num_samples == 0)
                    continue;

                // get the point in metric space
                double point[3];
                voxel_to_metric_space(x, y, z, len_x, len_y, len_z, voxel_size, point);

                // copy the point to the downsampled point cloud
                for(int i = 0; i < 3; i++) {
                    downsampled_point_cloud[downsampled_index*3 + i] = point[i];
                }
                downsampled_index++;
            }
        }
    }

    // free the normal distributions
    free(nd_array);

    // set the number of downsampled points
    *num_downsampled_points = downsampled_index;
}
