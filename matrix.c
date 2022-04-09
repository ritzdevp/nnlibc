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

Xarr* x_add(Xarr* arr1, Xarr* arr2){
    Xarr* res = Xinit(arr1->row, arr1->col);
    for (int i = 0; i < res->row; i++){
        for (int j = 0; j < res->col; j++){
            res->arr[i][j] = arr1->arr[i][j] + arr2->arr[i][j];
        }
    }
    return res;
}
