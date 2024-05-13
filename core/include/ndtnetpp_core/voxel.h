#ifndef VOXEL_H_
#define VOXEL_H_

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

#include <stdio.h>
#include <math.h>

enum direction_t {
    X_POS,
    X_NEG,
    Y_POS,
    Y_NEG,
    Z_POS,
    Z_NEG,
    DIRECTION_LEN
};

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
                        double max_x, double max_y, double max_z,
                        double min_x, double min_y, double min_z,
                        double *voxel_size,
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
int metric_to_voxel_space(double *point, double voxel_size,
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
                            double voxel_size, double *point);


/*! \brief Get the neighbor index in a given direction.
    \param index Index of the normal distribution in the array.
    \param len_x Number of voxels in the "x" dimension.
    \param len_y Number of voxels in the "y" dimension.
    \param len_z Number of voxels in the "z" dimension.
    \param direction Direction in the 3D space.
    \return Neighbor index.
*/
unsigned long get_neighbor_index(unsigned long index, int len_x, int len_y, int len_z, enum direction_t direction);

#endif // VOXEL_H_