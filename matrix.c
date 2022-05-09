//shape
//length
//transpose
//element wise product
//matmul, dot product
//add
//subtract
//mean
//concat
//reshape
//sort

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"


gsl_matrix* x_init(int row, int col){
    gsl_matrix* arr = gsl_matrix_calloc(row, col);
    return arr;
}

void x_free(gsl_matrix* arr){
    gsl_matrix_free(arr);
    return;
}

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

void x_print_shape(gsl_matrix* arr){
    size_t row = arr->size1;
    size_t col = arr->size2;
    printf("shape = (%lu, %lu)\n", row, col);
    return;
}

gsl_matrix* x_add(gsl_matrix* arr1, gsl_matrix* arr2){
    gsl_matrix* arr1_copy = gsl_matrix_calloc(arr1->size1, arr1->size2);
    gsl_matrix_memcpy(arr1_copy, arr1);
    //gsl_matrix_add does a[i][j] += b[i][j]
    //Hence, we make a copy first, so as to not alter the original array
    gsl_matrix_add(arr1_copy, arr2);
    // printf("Add success\n");
    return arr1_copy;
}

gsl_matrix* x_sub(gsl_matrix* arr1, gsl_matrix* arr2){
    gsl_matrix* arr1_copy = gsl_matrix_calloc(arr1->size1, arr1->size2);
    gsl_matrix_memcpy(arr1_copy, arr1);
    gsl_matrix_sub(arr1_copy, arr2);
    return arr1_copy;
}

gsl_matrix* x_scale(gsl_matrix* arr1, double k){
    gsl_matrix* arr1_copy = gsl_matrix_calloc(arr1->size1, arr1->size2);
    gsl_matrix_memcpy(arr1_copy, arr1);
    gsl_matrix_scale(arr1_copy, k);
    return arr1_copy;
}

gsl_matrix* x_transpose(gsl_matrix* arr1){
    gsl_matrix* arr1_trans = gsl_matrix_calloc(arr1->size2, arr1->size1);
    gsl_matrix_transpose_memcpy(arr1_trans, arr1);
    return arr1_trans;
}

// investigate
// https://stackoverflow.com/questions/67549023/why-is-the-gnu-scientific-library-matrix-multiplication-slower-than-numpy-matmul
gsl_matrix* x_dot(gsl_matrix* arr1, gsl_matrix* arr2){
    //https://www.gnu.org/software/gsl/doc/html/blas.html#examples

    //condition: c1 == r2
    if (arr1->size2 != arr2->size1){
        printf("Dim error: Can't dot (%lu, %lu) with (%lu, %lu)\n",
        arr1->size1, arr1->size2, arr2->size1, arr2->size2);
        return NULL;
    }
    
    //res shape: r1xc2
    // gsl_matrix* res = x_init(arr1->size1, arr2->size2);
    
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

//element wise multiplication
gsl_matrix* x_multiply(gsl_matrix* arr1, gsl_matrix* arr2){
    gsl_matrix* arr1_copy = gsl_matrix_calloc(arr1->size1, arr1->size2);
    gsl_matrix_memcpy(arr1_copy, arr1);
    gsl_matrix_mul_elements(arr1_copy, arr2);
    return arr1_copy;
}


//Fills the array of a Xarr struct object 
//With the elements stored in temp array
void x_fill(gsl_matrix* arr, void* temp){
    double (*array)[arr->size2] = temp;
    for (int i = 0; i < arr->size1; i++){
        for (int j = 0; j < arr->size2; j++){
            gsl_matrix_set(arr, i, j, array[i][j]);
        }
    }
    return;
}

gsl_matrix* x_log(gsl_matrix* arr){
    gsl_matrix* res = x_init(arr->size1, arr->size2);
    for (int i = 0; i < res->size1; i++){
        for (int j = 0; j < res->size2; j++){
            gsl_matrix_set(res, i, j, log(gsl_matrix_get(arr, i, j)));
        }
    }
    return res;
}

gsl_matrix* x_exp(gsl_matrix* arr){
    gsl_matrix* res = x_init(arr->size1, arr->size2);
    for (int i = 0; i < res->size1; i++){
        for (int j = 0; j < res->size2; j++){
            gsl_matrix_set(res, i, j, exp(gsl_matrix_get(arr, i, j)));
        }
    }
    return res;
}

gsl_matrix* x_ones(int row, int col){
    gsl_matrix* temp = x_init(row, col);
    gsl_matrix_set_all(temp, 1);
    return temp;
}

double x_mean(gsl_matrix* arr){
    int len = arr->size1 * arr->size2;
    return gsl_stats_mean(arr->data, 1, len);
}

//Note: in gsl_stats_mean(data, stride, number of elements)
//stride is the number of hops made during summation
gsl_matrix* x_mean_axis(gsl_matrix* arr, int axis){
    int row = arr->size1;
    int col = arr->size2;
    if (axis == 0){
        gsl_matrix* res = x_init(1, col);
        for (int i = 0; i < col; i++){
            //stride will be number of cols
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

gsl_matrix* x_copy(gsl_matrix* src){
    gsl_matrix* dest = gsl_matrix_calloc(src->size1, src->size2);
    gsl_matrix_memcpy(dest, src);
    return dest;
}

void zerofy_matrix(gsl_matrix* arr){
    for (int i = 0; i < arr->size1*arr->size2; i++){
        arr->data[i] = 0;
    }
    return;
}

gsl_matrix* x_broadcast_vector(gsl_matrix* arr, int rows){
    gsl_matrix* res = gsl_matrix_calloc(rows, arr->size2);
    for (int i = 0; i < res->size1; i++){
        for (int j = 0; j < res->size2; j++){
            gsl_matrix_set(res, i, j, gsl_matrix_get(arr, 0, j));
        }
    }
    return res;
}

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
    // printf("mean = %.15f\n", count/(double)(rows * cols));
}