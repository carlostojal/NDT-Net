#include <ndtnetpp_core/voxel.h>

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
                        int *len_x, int *len_y, int *len_z) {


    // calculate the lengths in each dimension
    double x_dim = max_x - min_x;
    double y_dim = max_y - min_y;
    double z_dim = max_z - min_z;

    // calculate the voxel size
    *voxel_size = cbrt((x_dim * y_dim * z_dim) / (double) num_desired_voxels);

    // calculate the number of voxels in each dimension
    *len_x = (int) x_dim / *voxel_size;
    *len_y = (int) y_dim / *voxel_size;
    *len_z = (int) z_dim / *voxel_size;
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
    point[0] = x_origin + voxel_x * voxel_size + voxel_size / 2;
    point[1] = y_origin + voxel_y * voxel_size + voxel_size / 2;
    point[2] = z_origin + voxel_z * voxel_size + voxel_size / 2;
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
    
    // calculate the neighbor index
    return index + direction_z * len_x * len_y + direction_y * len_x + direction_x;
}
