#include <ndnet_core/voxel.h>

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

void estimate_voxel_size(unsigned long num_desired_voxels,
                        double max_x, double max_y, double max_z,
                        double min_x, double min_y, double min_z,
                        double *voxel_size,
                        int *len_x, int *len_y, int *len_z,
                        double *x_offset, double *y_offset, double *z_offset) {


    // calculate the lengths in each dimension
    double x_dim = max_x - min_x;
    double y_dim = max_y - min_y;
    double z_dim = max_z - min_z;

    // calculate the voxel size
    double log_voxel_size = (log(x_dim) + log(y_dim) + log(z_dim) - log(num_desired_voxels)) / 3.0;
    *voxel_size = exp(log_voxel_size);

    // calculate the number of voxels in each dimension
    *len_x = ceil(x_dim / *voxel_size);
    *len_y = ceil(y_dim / *voxel_size);
    *len_z = ceil(z_dim / *voxel_size);

    // calculate the offsets
    // putting the center of the grid at the center of the metric space
    *x_offset = min_x;
    *y_offset = min_y;
    *z_offset = min_z;

    printf("Got grid with size [%d %d %d] and voxel size %f for lengths [%f %f %f] and offsets [%f %f %f].\n", *len_x, *len_y, *len_z, *voxel_size, 
                                                                                                                x_dim, y_dim, z_dim,
                                                                                                                *x_offset, *y_offset, *z_offset);
}

void estimate_voxel_grid(double max_x, double max_y, double max_z,
                        double min_x, double min_y, double min_z,
                        double voxel_size,
                        int *len_x, int *len_y, int *len_z,
                        double *x_offset, double *y_offset, double *z_offset) {

    // calculate the lengths in each dimension
    double x_dim = max_x - min_x;
    double y_dim = max_y - min_y;
    double z_dim = max_z - min_z;

    // calculate the number of voxels in each dimension
    *len_x = ceil(x_dim / voxel_size);
    *len_y = ceil(y_dim / voxel_size);
    *len_z = ceil(z_dim / voxel_size);

    // assign the offsets
    *x_offset = min_x;
    *y_offset = min_y;
    *z_offset = min_z;
}

int metric_to_voxel_space(double *point, double voxel_size,
                            int len_x, int len_y, int len_z,
                            double x_offset, double y_offset, double z_offset,
                            unsigned int *voxel_x, unsigned int *voxel_y, unsigned int *voxel_z) {

    // calculate the voxel indexes
    *voxel_x = (unsigned int) floor((point[0] - x_offset) / voxel_size);
    *voxel_y = (unsigned int) floor((point[1] - y_offset) / voxel_size);
    *voxel_z = (unsigned int) floor((point[2] - z_offset) / voxel_size);

    // check if the point is outside the grid
    if(*voxel_x >= len_x ||
        *voxel_y >= len_y ||
        *voxel_z >= len_z) {
        fprintf(stderr, "Point %f %f %f outside the grid! Got coordinates %d %d %d on a grid with len %d %d %d.\n", point[0], point[1], point[2]
                        , *voxel_x, *voxel_y, *voxel_z, len_x, len_y, len_z);
        return -1;
    }

    return 0;
}

void voxel_to_metric_space(unsigned int voxel_x, unsigned int voxel_y, unsigned int voxel_z,
                            int len_x, int len_y, int len_z,
                            double x_offset, double y_offset, double z_offset,
                            double voxel_size, double *point) {

    // calculate the point in metric space
    point[0] = (voxel_x * voxel_size) + (voxel_size / 2) + x_offset;
    point[1] = (voxel_y * voxel_size) + (voxel_size / 2) + y_offset;
    point[2] = (voxel_z * voxel_size) + (voxel_size / 2) + z_offset;
}

int get_neighbor_index(unsigned long index, unsigned int len_x, unsigned int len_y, unsigned int len_z, enum direction_t direction, unsigned long *neighbor_index) {

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
            return -2;
    }

    // update the positions
    unsigned int x, y, z;
    if(index_to_voxel_pos(index, len_x, len_y, len_z, &x, &y, &z) < 0) {
        fprintf(stderr, "Error getting neighbor voxel position!\n");
        return -3;
    }
    x += direction_x;
    y += direction_y;
    z += direction_z;

    if(x < 0 || x >= len_x ||
        y < 0 || y >= len_y ||
        z < 0 || z >= len_z) {
        // fprintf(stderr, "Neighbor voxel position outside the grid!\n");
        return -4;
    }

    // get the index
    if(voxel_pos_to_index(x, y, z, len_x, len_y, len_z, neighbor_index) < 0) {
        fprintf(stderr, "Error getting index from voxel position!\n");
        return -5;
    }

    return 0;
}

int voxel_pos_to_index(unsigned int voxel_x, unsigned int voxel_y, unsigned int voxel_z, int len_x, int len_y, int len_z, unsigned long *index) {

    if(voxel_x < 0 || voxel_x >= len_x ||
        voxel_y < 0 || voxel_y >= len_y ||
        voxel_z < 0 || voxel_z >= len_z) {
        fprintf(stderr, "Invalid voxel position for index!\n");
        return -1;
    }

    *index = voxel_z * len_x * len_y + voxel_y * len_x + voxel_x;

    return 0;
}

int index_to_voxel_pos(unsigned long index, int len_x, int len_y, int len_z, unsigned int *voxel_x, unsigned int *voxel_y, unsigned int *voxel_z) {

    if(index < 0 || index >= len_x * len_y * len_z) {
        fprintf(stderr, "Invalid index for voxel position!\n");
        return -1;
    }

    *voxel_z = index / (len_x * len_y);
    *voxel_y = (index % (len_x * len_y)) / len_x;
    *voxel_x = index % len_x;

    return 0;
}
