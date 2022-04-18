#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "activations.h"


gsl_matrix* act_identity(gsl_matrix* arr){
    gsl_matrix* res = x_init(arr->size1, arr->size2);
    gsl_matrix_memcpy(res, arr);
    return res;
}

gsl_matrix* act_sigmoid(gsl_matrix* arr){
    gsl_matrix* res = x_init(arr->size1, arr->size2);
    for (int i = 0; i < res->size1; i++){
        for (int j = 0; j < res->size2; j++){
            double temp = 1.0/(1 + exp(-gsl_matrix_get(arr, i, j)));
            gsl_matrix_set(res, i, j, temp);
        }
    }
    return res;
}

gsl_matrix* act_sigmoid_derivative(gsl_matrix* sigmoided_arr){
    size_t row = sigmoided_arr->size1;
    size_t col = sigmoided_arr->size2;
    gsl_matrix* allones = x_ones(row, col);
    gsl_matrix* res = x_multiply(sigmoided_arr, x_sub(allones, sigmoided_arr)); //sig*(1-sig)
    return res;
}

gsl_matrix* act_tanh(gsl_matrix* arr){
    gsl_matrix* res = x_init(arr->size1, arr->size2);
    for (int i = 0; i < res->size1; i++){
        for (int j = 0; j < res->size2; j++){
            double temp = tanh(gsl_matrix_get(arr, i, j));
            gsl_matrix_set(res, i, j, temp);
        }
    }
    return res;
}

gsl_matrix* act_tanh_derivative(gsl_matrix* tanhed_arr){
    size_t row = tanhed_arr->size1;
    size_t col = tanhed_arr->size2;
    gsl_matrix* allones = x_ones(row, col);
    gsl_matrix* res = x_sub(allones, x_multiply(tanhed_arr, tanhed_arr)); //1-tanh**2
    return res;
}

gsl_matrix* act_relu(gsl_matrix* arr){
    gsl_matrix* res = x_init(arr->size1, arr->size2);

    for (int i = 0; i < res->size1; i++){
        for (int j = 0; j < res->size2; j++){
            double temp = 0;
            if (gsl_matrix_get(arr, i, j) > 0){
                temp = gsl_matrix_get(arr, i, j);
            }
            gsl_matrix_set(res, i, j, temp);
        }
    }
    return res;
}

gsl_matrix* act_relu_derivative(gsl_matrix* relued_arr){
    gsl_matrix* res = x_init(relued_arr->size1, relued_arr->size2);
    for (int i = 0; i < res->size1; i++){
        for (int j = 0; j < res->size2; j++){
            double temp = 0;
            if (gsl_matrix_get(relued_arr, i, j) > 0){
                temp = 1;
            }
            gsl_matrix_set(res, i, j, temp);
        }
    }
    return res;
}

Activation* Act_init(char activation[10], int layer_index){
    Activation* act = malloc(sizeof(Activation));
    act->layer_index = layer_index;
    strcpy(act->activation, activation);
    return act;
}

gsl_matrix* act_forward(Activation* act, gsl_matrix* input){
    act->input = input;
    if (strcmp(act->activation, "relu") == 0){
        act->y = act_relu(input);
    }
    else if (strcmp(act->activation, "sigmoid") == 0){
        act->y = act_sigmoid(input);
    }
    else if (strcmp(act->activation, "tanh") == 0){
        act->y = act_tanh(input);
    }
    return act->y;
}

gsl_matrix* act_backward(Activation* act){
    if (strcmp(act->activation, "relu") == 0){
        act->derivative = act_relu_derivative(act->y);
    }
    else if (strcmp(act->activation, "sigmoid") == 0){
        act->derivative = act_sigmoid_derivative(act->y);
    }
    else if (strcmp(act->activation, "tanh") == 0){
        act->derivative = act_tanh_derivative(act->y);
    }
    return act->derivative; //f'(z)
}

void act_free(Activation* act){
    x_free(act->y);
    free(act);
    return;
}