#ifndef POINTCLOUDS_H_
#define POINTCLOUDS_H_

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

#include <float.h>


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
void get_pointcloud_limits(double *point_cloud, short point_dim, unsigned long num_points, 
                        double *max_x, double *max_y, double *max_z,
                        double *min_x, double *min_y, double *min_z);

#endif // POINTCLOUDS_H_