/**
 * @file matrix.c
 * @brief Numpy like wrapper over gsl matrix operations
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"

/**
 * @brief Initializes a gsl matrix
 * 
 * @param row Number of rows in the matrix
 * @param row Number of columns in the matrix
 * @return Pointer to gsl matrix
 */
gsl_matrix* x_init(int row, int col){
    gsl_matrix* arr = gsl_matrix_calloc(row, col);
    return arr;
}

/**
 * @brief Frees gsl matrix
 * 
 * @param arr Pointer to gsl matrix
 */
void x_free(gsl_matrix* arr){
    gsl_matrix_free(arr);
    return;
}

/**
 * @brief Pretty prints gsl matrix
 * 
 * @param arr Pointer to gsl matrix
 */
void x_print(gsl_matrix* arr){
    int row = arr->size1;
    int col = arr->size2;
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
            printf("%f ",gsl_matrix_get(arr, i, j));
        }
        printf("\n");
    }
}

/**
 * @brief Prints the matrix shape
 * 
 * @param arr Pointer to gsl matrix
 */
void x_print_shape(gsl_matrix* arr){
    size_t row = arr->size1;
    size_t col = arr->size2;
    printf("shape = (%lu, %lu)\n", row, col);
    return;
}

/**
 * @brief Adds two matrices of the same shape
 * 
 * @param arr1 Pointer to matrix
 * @param arr2 Pointer to matrix
 * @note gsl_matrix_add does a[i][j] += b[i][j]
 * Hence, we make a copy first, so as to not alter the original array 
 * @return Pointer to sum matrix
 */
gsl_matrix* x_add(gsl_matrix* arr1, gsl_matrix* arr2){
    gsl_matrix* arr1_copy = gsl_matrix_calloc(arr1->size1, arr1->size2);
    gsl_matrix_memcpy(arr1_copy, arr1);
    gsl_matrix_add(arr1_copy, arr2);
    return arr1_copy;
}

/**
 * @brief Subtracts two matrices of the same shape
 * arr1 - arr2
 * 
 * @param arr1 Pointer to matrix
 * @param arr2 Pointer to matrix
 * @return Pointer to subtraction result
 */
gsl_matrix* x_sub(gsl_matrix* arr1, gsl_matrix* arr2){
    gsl_matrix* arr1_copy = gsl_matrix_calloc(arr1->size1, arr1->size2);
    gsl_matrix_memcpy(arr1_copy, arr1);
    gsl_matrix_sub(arr1_copy, arr2);
    return arr1_copy;
}

/**
 * @brief Element wise multiplication by scalar
 * 
 * @param arr1 Pointer to matrix
 * @param k Scalar
 * @return Pointer to scaled matrix
 */
gsl_matrix* x_scale(gsl_matrix* arr1, double k){
    gsl_matrix* arr1_copy = gsl_matrix_calloc(arr1->size1, arr1->size2);
    gsl_matrix_memcpy(arr1_copy, arr1);
    gsl_matrix_scale(arr1_copy, k);
    return arr1_copy;
}

/**
 * @brief Transpose of matrix
 * 
 * @param arr1 Pointer to matrix
 * @return Pointer to transposed matrix
 */
gsl_matrix* x_transpose(gsl_matrix* arr1){
    gsl_matrix* arr1_trans = gsl_matrix_calloc(arr1->size2, arr1->size1);
    gsl_matrix_transpose_memcpy(arr1_trans, arr1);
    return arr1_trans;
}

/**
 * @brief Dot product of matrices
 * 
 * @param arr1 Pointer to matrix
 * @param arr2 Pointer to matrix
 * @return Pointer to dot product
 * @note Investigate https://stackoverflow.com/questions/67549023/why-is-the-gnu-scientific-library-matrix-multiplication-slower-than-numpy-matmul
 * @note //https://www.gnu.org/software/gsl/doc/html/blas.html#examples
 */
gsl_matrix* x_dot(gsl_matrix* arr1, gsl_matrix* arr2){
    /* condition: c1 == r2 */
    if (arr1->size2 != arr2->size1){
        printf("Dim error: Can't dot (%lu, %lu) with (%lu, %lu)\n",
        arr1->size1, arr1->size2, arr2->size1, arr2->size2);
        return NULL;
    }
    
    /* res shape: r1xc2 */
    
    gsl_matrix_view A = gsl_matrix_submatrix(arr1, 0, 0, arr1->size1, arr1->size2);
    gsl_matrix_view B = gsl_matrix_submatrix(arr2, 0, 0, arr2->size1, arr2->size2);
    double* res_arr = malloc(sizeof(double)*arr1->size1*arr2->size2);
    gsl_matrix_view res = gsl_matrix_view_array(res_arr, arr1->size1, arr2->size2);

    gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                  1.0, &A.matrix, &B.matrix,
                  0.0, &res.matrix);

    gsl_matrix* ans = x_init(arr1->size1, arr2->size2);
    int k = 0;
    for (int i = 0; i < ans->size1; i++){
        for (int j = 0; j < ans->size2; j++){
            gsl_matrix_set(ans, i, j, res_arr[k]);
            k++;
        }
    }

    return ans;
}

/**
 * @brief Element wise multiplication between matrices
 * 
 * @param arr1 Pointer to matrix
 * @param arr2 Pointer to matrix
 * @return Pointer to element wise product
 */
gsl_matrix* x_multiply(gsl_matrix* arr1, gsl_matrix* arr2){
    gsl_matrix* arr1_copy = gsl_matrix_calloc(arr1->size1, arr1->size2);
    gsl_matrix_memcpy(arr1_copy, arr1);
    gsl_matrix_mul_elements(arr1_copy, arr2);
    return arr1_copy;
}

/**
 * @brief Fills gsl matrix with elements in a 2D array
 * 
 * @param arr Pointer to matrix to be filled
 * @param temp Void pointer to 2D array
 */
void x_fill(gsl_matrix* arr, void* temp){
    double (*array)[arr->size2] = temp;
    for (int i = 0; i < arr->size1; i++){
        for (int j = 0; j < arr->size2; j++){
            gsl_matrix_set(arr, i, j, array[i][j]);
        }
    }
    return;
}

/**
 * @brief Element wise logarithm of matrix
 * 
 * @param arr Pointer to matrix
 * @return Pointer to log matrix
 */
gsl_matrix* x_log(gsl_matrix* arr){
    gsl_matrix* res = x_init(arr->size1, arr->size2);
    for (int i = 0; i < res->size1; i++){
        for (int j = 0; j < res->size2; j++){
            gsl_matrix_set(res, i, j, log(gsl_matrix_get(arr, i, j)));
        }
    }
    return res;
}

/**
 * @brief Element wise exponentiation of matrix
 * 
 * @param arr Pointer to matrix
 * @return Pointer to exp matrix
 */
gsl_matrix* x_exp(gsl_matrix* arr){
    gsl_matrix* res = x_init(arr->size1, arr->size2);
    for (int i = 0; i < res->size1; i++){
        for (int j = 0; j < res->size2; j++){
            gsl_matrix_set(res, i, j, exp(gsl_matrix_get(arr, i, j)));
        }
    }
    return res;
}

/**
 * @brief All ones matrix
 * 
 * @param row Number of rows
 * @param col Number of colums
 * @return Pointer to all ones matrix
 */
gsl_matrix* x_ones(int row, int col){
    gsl_matrix* temp = x_init(row, col);
    gsl_matrix_set_all(temp, 1);
    return temp;
}

/**
 * @brief Mean of entire matrix
 * 
 * @param arr Pointer to matrix
 * @return Mean of entire matrix
 */
double x_mean(gsl_matrix* arr){
    int len = arr->size1 * arr->size2;
    return gsl_stats_mean(arr->data, 1, len);
}

/**
 * @brief Mean of matrix along axis
 * 
 * @param arr Pointer to matrix
 * @param axis Axis along which mean is to be taken
 * @return Mean of matrix along axis
 * @note Note: in gsl_stats_mean(data, stride, number of elements)
 * stride is the number of hops made during summation
 */
gsl_matrix* x_mean_axis(gsl_matrix* arr, int axis){
    int row = arr->size1;
    int col = arr->size2;
    if (axis == 0){
        gsl_matrix* res = x_init(1, col);
        for (int i = 0; i < col; i++){
            /* stride will be number of cols */
            double mean_temp = gsl_stats_mean(arr->data+i, col, row);
            gsl_matrix_set(res, 0, i, mean_temp);
        }
        return res;
    }
    else if (axis == 1){
        gsl_matrix* res = x_init(1, row);
        for (int i = 0; i < row; i++){
            double mean_temp = gsl_stats_mean(arr->data + (i*col), 1, col);
            gsl_matrix_set(res, 0, i, mean_temp);
        }
        return res;
    }
    printf("Axis should be either 0 or 1.\n");
    return NULL;
}

/**
 * @brief Copy of a matrix
 * 
 * @param src Pointer to matrix
 * @return Copy of source matrix
 */
gsl_matrix* x_copy(gsl_matrix* src){
    gsl_matrix* dest = gsl_matrix_calloc(src->size1, src->size2);
    gsl_matrix_memcpy(dest, src);
    return dest;
}

/**
 * @brief Set all matrix elements to zero
 * 
 * @param arr Pointer to matrix
 */
void zerofy_matrix(gsl_matrix* arr){
    if (arr == NULL){
        return;
    }
    for (int i = 0; i < arr->size1*arr->size2; i++){
        arr->data[i] = 0;
    }
    return;
}

/**
 * @brief Broadcast vector into matrix
 * 
 * @param arr Pointer to matrix
 * @param Number of rows for broadcast matrix
 * @return Broadcasted matrix
 */
gsl_matrix* x_broadcast_vector(gsl_matrix* arr, int rows){
    gsl_matrix* res = gsl_matrix_calloc(rows, arr->size2);
    for (int i = 0; i < res->size1; i++){
        for (int j = 0; j < res->size2; j++){
            gsl_matrix_set(res, i, j, gsl_matrix_get(arr, 0, j));
        }
    }
    return res;
}

/**
 * @brief Get a row from matrix
 * 
 * @param arr Pointer to matrix
 * @param Index of row
 * @return Row
 */
gsl_matrix* get_row(gsl_matrix* arr, size_t row_index){
    
    /*
        This is a clever approach. Get row vector and then assign its
        data pointer to the data pointer of a matrix of shape 1xcols and return
        Have to test this later. But works most probably.
    */
    // gsl_vector* rowtemp = gsl_vector_calloc(arr->size2);
    // gsl_matrix_get_row(rowtemp, arr, row_index);
    // gsl_matrix* row = gsl_matrix_calloc(1, arr->size2);
    // row->data = rowtemp->data;

    gsl_matrix* row = gsl_matrix_calloc(1, arr->size2);
    for (int i = 0; i < arr->size2; i++){
        gsl_matrix_set(row, 0, i, gsl_matrix_get(arr, row_index, i));
    }
    return row;
}

/**
 * @brief Get index of maximum value
 * 
 * @param arr Pointer to 1xn matrix or row vector
 * @return Argmax
 */
size_t x_argmax_vector(gsl_matrix* arr){
    double max = gsl_matrix_get(arr, 0, 0);
    int index = 0;
    for (int i = 0; i < arr->size2; i++){
        if (gsl_matrix_get(arr, 0, i) > max){
            max = gsl_matrix_get(arr, 0, i);
            index = i;
        }
    }
    return index;
}

/**
 * @brief Xavier initialization for matrix
 * 
 * @param arr Pointer to matrix
 * @param rng Pointer to GSL random number generator
 */
void x_xavier_init(gsl_matrix* arr, gsl_rng * rng){
    size_t rows = arr->size1;
    size_t cols = arr->size2;
    double std_dev = sqrt(2.0/(double)((rows + cols)));
    double count = 0;
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            double init_val = gsl_ran_gaussian(rng, std_dev);
            count += init_val;
            gsl_matrix_set(arr, i, j, init_val);
        }
    }
}

/**
 * @brief Pretty print a sample of a matrix
 * 
 * @param arr Pointer to matrix
 */
void x_print_sample(gsl_matrix* arr){
    if (arr == NULL){
        printf("NULL parameter\n");
        return;
    }
    int limit = 2;
    if (arr->size1 == 1){
        limit = 1;
    }
    int once = 0;
    for (int i = 0; i < arr->size1; i++){
        for (int j = 0; j < arr->size2; j++){
            if (gsl_matrix_get(arr, i, j) != 0){
                once = 1;
                break;
            }
        }
    }
    if (!once){
        printf("ALL ZEROS!\n");
    }
    for (int i = 0; i < limit; i++){
        for (int j = 0; j < 4;  j++){
            printf("%.6f ", gsl_matrix_get(arr, i, j));
        }
        printf("\n");
    }
}