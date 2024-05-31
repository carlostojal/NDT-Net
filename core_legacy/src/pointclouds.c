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

double maxf(double a, double b) {
    return a > b ? a : b;
}

double minf(double a, double b) {
    return a < b ? a : b;
}

double absf(double n) {
    return n < 0 ? -n : n;
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

        *max_x = maxf(point_cloud[i*point_dim], *max_x);
        *min_x = minf(point_cloud[i*point_dim], *min_x);

        *max_y = maxf(point_cloud[i*point_dim + 1], *max_y);
        *min_y = minf(point_cloud[i*point_dim + 1], *min_y);

        *max_z = maxf(point_cloud[i*point_dim + 2], *max_z);
        *min_z = minf(point_cloud[i*point_dim + 2], *min_z);        
    }

    // printf("Limits [%f %f], [%f %f], [%f %f]\n", *min_x, *max_x, *min_y, *max_y, *min_z, *max_z);
}