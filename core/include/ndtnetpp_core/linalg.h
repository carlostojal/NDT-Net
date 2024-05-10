#ifndef LINALG_H_
#define LINALG_H_

#include <stdio.h>
#include <stdlib.h>

/*! \brief Multiply two matrices.
    \param mat1 Pointer to the first matrix. Row-major order.
    \param mat2 Pointer to the second matrix. Row-major order.
    \param result Pointer to the result matrix. Row-major order.
    \param rows1 Number of rows in the first matrix.
    \param cols1 Number of columns in the first matrix.
    \param rows2 Number of rows in the second matrix.
    \param cols2 Number of columns in the second matrix.

*/
void matmul(float *mat1, float *mat2, float *result, 
            int rows1, int cols1, int rows2, int cols2);

/*! \brief Subtract two matrices.
    \param mat1 Pointer to the first matrix. Row-major order.
    \param mat2 Pointer to the second matrix. Row-major order.
    \param result Pointer to the result matrix. Row-major order.
    \param len Length of the matrices.
*/
void matsub(float *mat1, float *mat2, float *result, int rows, int cols);

/*! \brief Add two matrices.
    \param mat1 Pointer to the first matrix. Row-major order.
    \param mat2 Pointer to the second matrix. Row-major order.
    \param result Pointer to the result matrix. Row-major order.
    \param len Length of the matrices.
*/
void matadd(float *mat1, float *mat2, float *result, int rows, int cols);

/*! \brief Dot product of two vectors.
    \param vec1 Pointer to the first vector.
    \param vec2 Pointer to the second vector.
    \param result Pointer to the result vector.
    \param len Length of the vectors.
*/
void dot_product(float *vec1, float *vec2, float *result, int len);

/*! \brief Trace of a matrix (sum of the diagonal).
    \param matrix Pointer to the matrix. Row-major order.
    \param result Pointer to the result.
    \param len Length of the matrix.
*/
void trace(float *matrix, float *result, int len);

/*! \brief Transpose a matrix.
    \param matrix Pointer to the matrix. Row-major order.
    \param result Pointer to the result matrix. Row-major order.
    \param rows Number of rows in the matrix.
    \param cols Number of columns in the matrix.
*/
void transpose(float *matrix, float *result, int rows, int cols);

/*! \brief Calculate the determinant of a matrix. Using recursion and the Laplace expansion method.
    \param matrix Pointer to the matrix. Row-major order.
    \param result Pointer to the result.
    \param len Length of the matrix.
*/
int det(float *matrix, float *result, int len);

/*! \brief Calculate the inverse of a matrix. Using the determinant and the adjugate matrix.
    \param matrix Pointer to the matrix. Row-major order.
    \param result Pointer to the result matrix. Row-major order.
    \param len Length of the matrix.
*/
int inv(float *matrix, float *result, int len);

#endif // LINALG_H_
