#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "activations.h"


Xarr* act_identity(Xarr* arr){
    Xarr* res = Xinit(arr->row, arr->col);
    for (int i = 0; i < arr->row; i++){
        for (int j = 0; j < arr->col; j++){
            res->arr[i][j] = arr->arr[i][j];
        }
    }
    return res;
}

Xarr* act_sigmoid(Xarr* arr){
    Xarr* res = Xinit(arr->row, arr->col);
    for (int i = 0; i < arr->row; i++){
        for (int j = 0; j < arr->col; j++){
            double temp = 1.0/(1 + exp(-arr->arr[i][j]));
            res->arr[i][j] = temp;
        }
    }
    return res;
}

Xarr* act_tanh(Xarr* arr){
    Xarr* res = Xinit(arr->row, arr->col);
    for (int i = 0; i < arr->row; i++){
        for (int j = 0; j < arr->col; j++){
            double temp = tanh(arr->arr[i][j]);
            res->arr[i][j] = temp;
        }
    }
    return res;
}

Xarr* act_relu(Xarr* arr){
    Xarr* res = Xinit(arr->row, arr->col);
    for (int i = 0; i < arr->row; i++){
        for (int j = 0; j < arr->col; j++){
            double temp = 0;
            if (arr->arr[i][j] > 0){
                temp = arr->arr[i][j];
            }
            res->arr[i][j] = temp;
        }
    }
    return res;
}