#include <ndtnetpp_core/linalg.h>

void matmul(float *mat1, float *mat2, float *result, 
            int rows1, int cols1, int rows2, int cols2) {
    
    // Check if the matrices can be multiplied
    if (cols1 != rows2) {
        fprintf(stderr, "Incompatible matrix shapes!\n");
        return;
    }

    // Perform matrix multiplication
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            result[i * cols2 + j] = 0;
            for (int k = 0; k < cols1; k++) {
                result[i * cols2 + j] += mat1[i * cols1 + k] * mat2[k * cols2 + j];
            }
        }
    }
}

void dot_product(float *vec1, float *vec2, float *result, int len) {
    *result = 0;
    for (int i = 0; i < len; i++) {
        *result += vec1[i] * vec2[i];
    }
}

void trace(float *matrix, float *result, int len) {
    *result = 0;
    for (int i = 0; i < len; i++) {
        *result += matrix[i * len + i];
    }
}

void transpose(float *matrix, float *result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j * rows + i] = matrix[i * cols + j];
        }
    }
}

void det(float *matrix, float *result, int len) {
    if (len == 1) {
        *result = matrix[0];
        return;
    }

    if (len == 2) {
        *result = matrix[0] * matrix[3] - matrix[1] * matrix[2];
        return;
    }

    float *submatrix = (float *)malloc((len - 1) * (len - 1) * sizeof(float));
    float subdet;
    *result = 0;
    for (int i = 0; i < len; i++) {
        int subi = 0;
        for (int j = 1; j < len; j++) {
            int subj = 0;
            for (int k = 0; k < len; k++) {
                if (k == i) {
                    continue;
                }
                submatrix[subi * (len - 1) + subj] = matrix[j * len + k];
                subj++;
            }
            subi++;
        }
        det(submatrix, &subdet, len - 1);
        *result += (i % 2 == 0 ? 1 : -1) * matrix[i] * subdet;
    }
    free(submatrix);
}
