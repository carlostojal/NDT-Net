#include <ndtnetpp_core/pointclouds.h>

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