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
#include "matrix.h"


Xarr * Xinit(int row, int col){
    Xarr * xarr = malloc(sizeof(Xarr));
    xarr->row = row;
    xarr->col = col;

    xarr->arr = (double **)malloc(row * sizeof(double *));
    for (int i = 0; i < row; i++){
        xarr->arr[i] = malloc(col * sizeof(double));
    }

    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
            xarr->arr[i][j] = 0;
        }
    }
    return xarr;
}

void Xfree(Xarr* arr){
    for (int i = 0; i < arr->row; i++){
        free(arr->arr[i]);
    }
    free(arr->arr);
    free(arr);
}

void x_print(Xarr* arr){
    for (int i = 0; i < arr->row; i++){
        for (int j = 0; j < arr->col; j++){
            printf("%f ",arr->arr[i][j]);
        }
        printf("\n");
    }
}

Xarr* x_add(Xarr* arr1, Xarr* arr2){
    Xarr* res = Xinit(arr1->row, arr1->col);
    for (int i = 0; i < res->row; i++){
        for (int j = 0; j < res->col; j++){
            res->arr[i][j] = arr1->arr[i][j] + arr2->arr[i][j];
        }
    }
    return res;
}

Xarr* x_sub(Xarr* arr1, Xarr* arr2){
    Xarr* res = Xinit(arr1->row, arr1->col);
    for (int i = 0; i < res->row; i++){
        for (int j = 0; j < res->col; j++){
            res->arr[i][j] = arr1->arr[i][j] - arr2->arr[i][j];
        }
    }
    return res;
}

Xarr* x_scale(Xarr* arr1, double k){
    Xarr* res = Xinit(arr1->row, arr1->col);
    for (int i = 0; i < res->row; i++){
        for (int j = 0; j < res->col; j++){
            res->arr[i][j] = k * arr1->arr[i][j];
        }
    }
    return res;
}

Xarr* x_transpose(Xarr* arr1){
    Xarr* res = Xinit(arr1->col, arr1->row);
    for (int i = 0; i < arr1->row; i++){
        for (int j = 0; j < arr1->col; j++){
            res->arr[j][i] = arr1->arr[i][j];
        }
    }
    return res;
}

Xarr* x_dot(Xarr* arr1, Xarr* arr2){
    //condition: c1 == r2
    if (arr1->col != arr2->row){
        printf("Dim error: Can't dot (%d, %d) with (%d, %d)\n",
        arr1->row, arr1->col, arr2->row, arr2->col);
        return NULL;
    }
    
    //res shape: r1xc2

    Xarr* res = Xinit(arr1->row, arr2->col);
    for (int row_1 = 0; row_1 < arr1->row; row_1++){
        for (int col_2 = 0; col_2 < arr2->col; col_2++){
            double temp_sum = 0;
            for (int i = 0; i < arr1->col; i++){
                temp_sum += arr1->arr[row_1][i] * arr2->arr[i][col_2];
            }
            res->arr[row_1][col_2] = temp_sum;
        }
    }
    return res;
}