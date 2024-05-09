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

void get_pointcloud_limits(float *point_cloud, short point_dim, unsigned long num_points,
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

void estimate_voxel_size(unsigned long num_desired_voxels,
                        float max_x, float max_y, float max_z,
                        float min_x, float min_y, float min_z,
                        float *voxel_size,
                        int *len_x, int *len_y, int *len_z) {


    // calculate the lengths in each dimension
    float x_dim = max_x - min_x;
    float y_dim = max_y - min_y;
    float z_dim = max_z - min_z;

    // calculate the voxel size
    *voxel_size = (float) num_desired_voxels / x_dim;
    *voxel_size /= y_dim;
    *voxel_size /= z_dim;

    *voxel_size = floor(*voxel_size);
}

int metric_to_voxel_space(float *point, float voxel_size,
                            int len_x, int len_y, int len_z,
                            unsigned int *voxel_x, unsigned int *voxel_y, unsigned int *voxel_z) {

    // the center voxel of the grid is at metric (0, 0, 0)

    // find the origin of the grid in metric space
    float x_origin, y_origin, z_origin;
    x_origin = -((float) len_x / 2) * voxel_size;
    y_origin = -((float) len_y / 2) * voxel_size;
    z_origin = -((float) len_z / 2) * voxel_size;

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
                            float voxel_size, float *point) {

    // the center voxel of the grid is at metric (0, 0, 0)

    // find the origin of the grid in metric space
    float x_origin, y_origin, z_origin;
    x_origin = -((float) len_x / 2) * voxel_size;
    y_origin = -((float) len_y / 2) * voxel_size;
    z_origin = -((float) len_z / 2) * voxel_size;

    // calculate the point in metric space
    point[0] = x_origin + voxel_x * voxel_size;
    point[1] = y_origin + voxel_y * voxel_size;
    point[2] = z_origin + voxel_z * voxel_size;
}

void *pcl_worker(void *arg) {

    // TODO
}


int estimate_ndt(float *point_cloud, unsigned long num_points, float voxel_size,
                    int len_x, int len_y, int len_z) {


    // create an array of normal distributions, one per voxel
    struct normal_distribution_t *nd_array = (struct normal_distribution_t *) malloc(len_x * len_y * len_z * sizeof(struct normal_distribution_t));
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

    // free the array of normal distributions
    free(nd_array);

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
