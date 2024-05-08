#include <stdio.h>
#include <pthread.h>

/*! \brief Print a matrix to the standard output. 
    \param matrix Pointer to the matrix. Row-major order.
    \param rows Number of rows in the matrix.
    \param cols Number of columns in the matrix.
*/
void print_matrix(float *matrix, int rows, int cols);
