#ifndef KULLBACK_LEIBLER_H_
#define KULLBACK_LEIBLER_H_

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

#include <ndtnetpp_core/ndt.h>

#include <stdio.h>
#include <errno.h>
#include <string.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <omp.h>

#include <ndtnetpp_core/voxel.h>

struct kl_divergence_t {
    double divergence; // divergence value
    struct normal_distribution_t *p; // pointer to the first normal distribution
    struct normal_distribution_t *q; // pointer to the second normal distribution
};

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Compute the multivariate Kullback-Leibler divergence between two normal distributions.
    \param p Pointer to the first normal distribution.
    \param q Pointer to the second normal distribution.
    \param divergence Pointer to the divergence value. Will be overwritten.
    \return 0 if successful, -1 otherwise.
*/
int kl_divergence(struct normal_distribution_t *p, struct normal_distribution_t *q, double *divergence);

#ifdef __cplusplus
}
#endif

#endif // KULLBACK_LEIBLER_H_