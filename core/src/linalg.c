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

int det(float *matrix, float *result, int len) {
    if (len == 1) {
        *result = matrix[0];
        return;
    }

    if (len == 2) {
        *result = matrix[0] * matrix[3] - matrix[1] * matrix[2];
        return;
    }

    float *submatrix = (float *)malloc((len - 1) * (len - 1) * sizeof(float));
    if(submatrix == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        return -1;
    }
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

int inv(float *matrix, float *result, int len) {

    // calculate the determinant
    float determinant;
    if(det(matrix, &determinant, len) < 0) {
        fprintf(stderr, "Error computing the determinant!\n");
        return -1;
    }
    if (determinant == 0) {
        fprintf(stderr, "Matrix is singular!\n");
        return -2;
    }

    // calculate the adjugate matrix
    float *adjugate = (float *)malloc(len * len * sizeof(float));
    if(adjugate == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        return -3;
    }
    float *submatrix = (float *)malloc((len - 1) * (len - 1) * sizeof(float));  
    if(submatrix == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        free(adjugate);
        return -4;
    }
    float subdet;
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < len; j++) {
            int subi = 0;
            for (int k = 0; k < len; k++) {
                if (k == i) {
                    continue;
                }
                int subj = 0;
                for (int l = 0; l < len; l++) {
                    if (l == j) {
                        continue;
                    }
                    submatrix[subi * (len - 1) + subj] = matrix[k * len + l];
                    subj++;
                }
                subi++;
            }
            det(submatrix, &subdet, len - 1);
            adjugate[j * len + i] = (i + j) % 2 == 0 ? subdet : -subdet;
        }
    }
    free(submatrix);

    // calculate the inverse
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < len; j++) {
            result[i * len + j] = adjugate[i * len + j] / determinant;
        }
    }
    free(adjugate);
    return 0;
}

void matsub(float *mat1, float *mat2, float *result, int rows, int cols) {

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < rows; j++) {
            result[i * cols + j] = mat1[i * cols + j] - mat2[i * cols + j];
        }
    }
}

void matadd(float *mat1, float *mat2, float *result, int rows, int cols) {

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < rows; j++) {
            result[i * cols + j] = mat1[i * cols + j] + mat2[i * cols + j];
        }
    }
}
