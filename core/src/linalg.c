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
