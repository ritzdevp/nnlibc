/**
 * @file layers.c
 * @brief Contains Linear layer methods
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "layers.h"

/**
 * @brief Initializes linear layer
 * 
 * @param input_size Input dimension
 * @param output_size Output dimension
 * @param layer_index Index of this layer in the network
 * @param rng Pointer to gsl random number generator
 * @return Pointer to Lineary layer
 */
Linear* linear_init(int input_size, int output_size, int layer_index, gsl_rng * rng){
    Linear* linear_layer = malloc(sizeof(Linear));

    linear_layer->input_size = input_size;
    linear_layer->output_size = output_size;
    linear_layer->layer_index = layer_index;

    linear_layer->W = x_init(input_size, output_size);
    linear_layer->b = x_init(1, output_size);

    /* Xavier Initialization of layer weights */
    x_xavier_init(linear_layer->W, rng);
    
    return linear_layer;
}

/**
 * @brief Linear layer forward method
 * 
 * @param input Input matrix to linear layer
 * @param linear_layer Pointer to linear layer
 * @return Pointer to linear layer output
 */
gsl_matrix* forward(gsl_matrix* input, Linear* linear_layer){
    
    linear_layer->x = input;

    // Works only for batchsize == 1
    // linear_layer->z = x_add(x_dot(input, linear_layer->W), linear_layer->b);
    
    /* z = x.dot(W) + b */
    /* broadcasting is required for batch_size > 1 */
    linear_layer->z = x_add(x_dot(input, linear_layer->W), x_broadcast_vector(linear_layer->b, input->size1));
    
    return linear_layer->z;
}

/**
 * @brief Linear layer backward method
 * 
 * @param linear_layer Pointer to linear layer
 * @param dLdz Gradient of loss wrt linear layer output
 * @return Pointer to gradient of loss wrt linear layer input
 */
gsl_matrix* backward(Linear* linear_layer, gsl_matrix* dLdz){
    double batch_size = dLdz->size1;
    
    /* dldW = (1/bs) * (x.T.dot(dLdz)) */
    linear_layer->dLdW = x_scale(x_dot(x_transpose(linear_layer->x), dLdz), (double)(1.0/batch_size));
    
    /* dLdb = mean(dLdz, axis=0, keepdims=True) */
    linear_layer->dLdb = x_mean_axis(dLdz, 0);

    /* dLdx = dLdz.dot(W.T) */
    gsl_matrix* dLdx = x_dot(dLdz, x_transpose(linear_layer->W));
    
    return dLdx;
}

/**
 * @brief Linear layer free method
 * 
 * @param linear_layer Pointer to linear layer
 */
void linear_free(Linear* linear_layer){
    x_free(linear_layer->W);
    x_free(linear_layer->b);

    if (linear_layer->z != NULL){
        x_free(linear_layer->z);
    }
    free(linear_layer);
    return;
}