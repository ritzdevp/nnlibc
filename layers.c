#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "layers.h"

Linear* linear_init(int input_size, int output_size, int layer_index){
    Linear* linear_layer = malloc(sizeof(Linear));

    linear_layer->input_size = input_size;
    linear_layer->output_size = output_size;
    linear_layer->layer_index = layer_index;

    linear_layer->W = x_init(input_size, output_size);
    linear_layer->b = x_init(1, output_size);
    
    return linear_layer;
}

gsl_matrix* forward(gsl_matrix* input, Linear* linear_layer){
    //z = x.dot(W) + b
    linear_layer->x = input;
    linear_layer->z = x_add(x_dot(input, linear_layer->W), linear_layer->b);
    return linear_layer->z;
}

gsl_matrix* backward(Linear* linear_layer, gsl_matrix* dLdz){
    double batch_size = dLdz->size1; //bs
    
    //dldW = (1/bs) * (x.T.dot(dLdz))
    linear_layer->dLdW = x_scale(x_dot(x_transpose(linear_layer->x), dLdz), (double)(1/batch_size));
    
    //dLdb = mean(dLdz, axis=0, keepdims=True)
    linear_layer->dLdb = x_mean_axis(dLdz, 0);

    //dLdx = dLdz.dot(W.T)
    gsl_matrix* dLdx = x_dot(dLdz, x_transpose(linear_layer->W));

    return dLdx;
}

void linear_free(Linear* linear_layer){
    x_free(linear_layer->W);
    x_free(linear_layer->b);

    if (linear_layer->z != NULL){
        x_free(linear_layer->z);
    }
    free(linear_layer);
    return;
}