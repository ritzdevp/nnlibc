/**
 * @file activations.c
 * @brief Contains activation functions, their
 * derivatives, forward and backward methods.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "activations.h"

/**
 * @brief Identity activation function.
 * 
 * @param arr Pointer to matrix
 * @return The copy of the matrix
 * itself.
 */
gsl_matrix* act_identity(gsl_matrix* arr){
    gsl_matrix* res = x_init(arr->size1, arr->size2);
    gsl_matrix_memcpy(res, arr);
    return res;
}

/**
 * @brief Derivative of identity activation function.
 * 
 * @param arr Pointer to matrix
 * @return All ones matrix
 */
gsl_matrix* act_identity_derivative(gsl_matrix* arr){
    gsl_matrix* res = x_ones(arr->size1, arr->size2);
    return res;
}

/**
 * @brief Sigmoid activation function
 * 
 * @param arr Pointer to matrix
 * @return Pointer to matrix after applying Sigmoid
 */
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

/**
 * @brief Derivative of sigmoid activation function
 * 
 * @param sigmoided_arr Pointer to matrix
 * @return Pointer to matrix derivative
 */
gsl_matrix* act_sigmoid_derivative(gsl_matrix* sigmoided_arr){
    size_t row = sigmoided_arr->size1;
    size_t col = sigmoided_arr->size2;
    gsl_matrix* allones = x_ones(row, col);
    
    //sig*(1-sig)
    gsl_matrix* res = x_multiply(sigmoided_arr, x_sub(allones, sigmoided_arr));
    return res;
}

/**
 * @brief Tanh activation function
 * 
 * @param arr Pointer to matrix
 * @return Pointer to matrix afer applying Tanh
 */
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

/**
 * @brief Derivative of Tanh activation function
 * 
 * @param tanhed_arr Pointer to matrix
 * @return Pointer to matrix derivative
 */
gsl_matrix* act_tanh_derivative(gsl_matrix* tanhed_arr){
    size_t row = tanhed_arr->size1;
    size_t col = tanhed_arr->size2;
    gsl_matrix* allones = x_ones(row, col);
    gsl_matrix* res = x_sub(allones, x_multiply(tanhed_arr, tanhed_arr)); //1-tanh**2
    return res;
}

/**
 * @brief ReLu activation function
 * 
 * @param arr Pointer to matrix
 * @return Pointer to matrix afer applying ReLu
 */
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

/**
 * @brief Derivative of ReLu activation function
 * 
 * @param relued_arr Pointer to matrix
 * @return Pointer to matrix derivative
 */
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

/**
 * @brief Initializes activation layer
 * 
 * @param activation String that specifies type of activation
 * "relu", "sigmoid", "tanh", "identity"
 * @param layer_index The index of this layer in the network
 * @return Pointer to Activation layer
 */
Activation* Act_init(char activation[10], int layer_index){
    Activation* act = malloc(sizeof(Activation));
    act->layer_index = layer_index;
    strcpy(act->activation, activation);
    return act;
}

/**
 * @brief Forward method for Activation layer
 * 
 * @param act Activation layer
 * @param input The input to the layer
 * @return Pointer to activation output 
 */
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
    else if (strcmp(act->activation, "identity") == 0){
        act->y = act_identity(input);
    }
    return act->y;
}

/**
 * @brief Backward method for Activation layer
 * 
 * @param act Activation layer
 * @return Pointer to activation derivative 
 */
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
    else if (strcmp(act->activation, "identity") == 0){
        act->derivative = act_identity_derivative(act->y);
    }
    return act->derivative; //f'(z)
}

/**
 * @brief Frees activation layer
 * 
 * @param act Activation layer
 */
void act_free(Activation* act){
    x_free(act->y);
    free(act);
    return;
}