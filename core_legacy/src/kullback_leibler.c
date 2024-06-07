#include <ndtnetpp_core/kullback_leibler.h>

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

int kl_divergence(struct normal_distribution_t *p, struct normal_distribution_t *q, double *divergence) {

    // calculate the divergence between two normal distributions
    // the divergence is the multivariate Kullback-Leibler divergence

    *divergence = 0;

    /*
    printf("CALCULATING DIVERGENCE BETWEEN %lu AND %lu\n", p->index, q->index);
    print_nd(*p);
    print_nd(*q);
    printf("--------------------------------\n");
    */

    if(p->num_samples <= 1 || q->num_samples <= 1) {
        // fprintf(stderr, "Not enough samples!\n");
        return -1;
    }

    // create GSL matrices from the covariance matrices
    gsl_matrix_view p_covariance = gsl_matrix_view_array(p->covariance, 3, 3);
    gsl_matrix_view q_covariance = gsl_matrix_view_array(q->covariance, 3, 3);

    // make the LU decomposition of the covariance matrices
    gsl_matrix *p_LU = gsl_matrix_alloc(3, 3);
    gsl_matrix *q_LU = gsl_matrix_alloc(3, 3);
    gsl_permutation *p_permutation = gsl_permutation_alloc(3);
    gsl_permutation *q_permutation = gsl_permutation_alloc(3);
    int p_signum, q_signum;
    gsl_linalg_LU_decomp(&(p_covariance.matrix), p_permutation, &p_signum);
    gsl_linalg_LU_decomp(&(q_covariance.matrix), q_permutation, &q_signum);
    gsl_permutation_free(p_permutation);

    // calculate the determinant of the covariance matrices
    double p_det = gsl_linalg_LU_det(&(p_covariance.matrix), p_signum);
    double q_det = gsl_linalg_LU_det(&(q_covariance.matrix), q_signum);

    // check if the determinants are zero
    if(p_det == 0 || q_det == 0) {
        // fprintf(stderr, "The covariance matrix is singular!\n");
        return -2;
    }
    // check if the matrices are invertible by the rank
    if(gsl_linalg_LU_sgndet(&(p_covariance.matrix), p_signum) == 0 || gsl_linalg_LU_sgndet(&(q_covariance.matrix), q_signum) == 0) {
        // fprintf(stderr, "The \"p\" covariance matrix is singular!\n");
        return -2;
    }
    if(gsl_linalg_LU_sgndet(&(q_covariance.matrix), q_signum) == 0 || gsl_linalg_LU_sgndet(&(q_covariance.matrix), q_signum) == 0) {
        // fprintf(stderr, "The \"q\"covariance matrix is singular!\n");
        return -2;
    }

    // calculate the difference between the means
    gsl_matrix *mean_diff = gsl_matrix_alloc(3, 1); // allocate the mean difference vector
    gsl_matrix_view p_mean = gsl_matrix_view_array(p->mean, 3, 1);
    gsl_matrix_view q_mean = gsl_matrix_view_array(q->mean, 3, 1);
    gsl_matrix_memcpy(mean_diff, &q_mean.matrix); // copy the p mean to the difference
    gsl_matrix_sub(mean_diff, &p_mean.matrix); // subtract the q mean from the difference
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
    double first_part_result = 0;
    // convert the first part and mean difference to a GSL vector
    gsl_vector_view first_part_view = gsl_vector_view_array(first_part->data, 3);
    gsl_vector_view mean_diff_view = gsl_vector_view_array(mean_diff->data, 3);
    gsl_blas_ddot(&(first_part_view.vector), &(mean_diff_view.vector), &first_part_result); // calculate the dot product of the first part and the mean difference

    // calculate the divergence
    *divergence = 0.5 * (first_part_result + trace - log(q_det/p_det) - 3);

    // free the allocated memory
    gsl_matrix_free(p_LU);
    gsl_matrix_free(q_LU);
    gsl_matrix_free(mean_diff);
    gsl_matrix_free(mean_diff_transpose);
    gsl_matrix_free(q_inverse);
    gsl_matrix_free(trace_matrix);
    gsl_matrix_free(first_part);

    return 0;
}

int calculate_kl_divergences(struct normal_distribution_t *nd_array,
                            unsigned int len_x, unsigned int len_y, unsigned int len_z,
                            unsigned long *num_valid_nds,
                            struct kl_divergence_t *kl_divergences, unsigned long *num_kl_divergences) {

    // initialize the counts to zero
    *num_valid_nds = 0;
    *num_kl_divergences = 0;

    // calculate the divergences between each pair of neighboring distributions
    // also, count the valid normal distributions
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
                (*num_valid_nds)++;

                // calculate the divergence between the current voxel and the neighbors in each direction
                for(short i = 0; i < DIRECTION_LEN; i++) {

                    // get the neighbor index
                    unsigned long neighbor_index;
                    if(get_neighbor_index(index, len_x, len_y, len_z, i, &neighbor_index) == -4) { // neighbor out of bounds
                        continue;
                    } else if (neighbor_index < 0) {
                        fprintf(stderr, "Error getting neighbor index!\n");
                        return -2;
                    }

                    // verify if the other voxel has samples
                    if(nd_array[neighbor_index].num_samples == 0)
                        continue;
                    
                    // calculate the divergence between the distributions
                    double div = 0;
                    if(kl_divergence(&nd_array[index], &nd_array[neighbor_index], &div) == -2) {
                        // the q covariance matrix is singular
                        continue;
                    }

                    // insert the divergence in the ordered array
                    unsigned long j = 0;
                    while(j < *num_kl_divergences) {
                        if(kl_divergences[j].divergence < div)
                            break;
                        j++;
                    }
                    // shift the divergences to the right
                    for(unsigned long k = *num_kl_divergences; k > j; k--) {
                        kl_divergences[k] = kl_divergences[k-1];
                    }
                    // insert the divergence
                    kl_divergences[j].divergence = div;
                    kl_divergences[j].p = &nd_array[index];
                    kl_divergences[j].q = &nd_array[neighbor_index];
                    (*num_kl_divergences)++;
                }
            }
        }
    }

    return 0;
}

void free_kl_divergences(struct kl_divergence_t *kl_divergences) {
    free(kl_divergences);
    kl_divergences = NULL;
}
