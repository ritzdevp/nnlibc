#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "layers.h"

Linear* linear_init(int input_size, int output_size, int layer_index, gsl_rng * rng){
    Linear* linear_layer = malloc(sizeof(Linear));

    linear_layer->input_size = input_size;
    linear_layer->output_size = output_size;
    linear_layer->layer_index = layer_index;

    linear_layer->W = x_init(input_size, output_size);
    linear_layer->b = x_init(1, output_size);

    x_xavier_init(linear_layer->W, rng);
    
    return linear_layer;
}

gsl_matrix* forward(gsl_matrix* input, Linear* linear_layer){
    //z = x.dot(W) + b
    // printf("Input for layer %.15f\n", x_mean(input));
    linear_layer->x = input;

    //Works only for batchsize == 1
    // linear_layer->z = x_add(x_dot(input, linear_layer->W), linear_layer->b);
    
    //Need broadcast if doing with batch_size > 1
    linear_layer->z = x_add(x_dot(input, linear_layer->W), x_broadcast_vector(linear_layer->b, input->size1));
    
    return linear_layer->z;
}

gsl_matrix* backward(Linear* linear_layer, gsl_matrix* dLdz){
    double batch_size = dLdz->size1; //bs
    
    //dldW = (1/bs) * (x.T.dot(dLdz))
    linear_layer->dLdW = x_scale(x_dot(x_transpose(linear_layer->x), dLdz), (double)(1.0/batch_size));
    // printf("dLdW layer = %d\n", linear_layer->layer_index);
    // x_print_sample(linear_layer->dLdW);
    // if (linear_layer->layer_index == 0){
    //     for (int i = 0; i < linear_layer->dLdW->size1; i++){
    //         for (int j = 0; j < linear_layer->dLdW->size2; j++){
    //             if (gsl_matrix_get(linear_layer->dLdW, i, j) != 0){
    //                 printf("VOILA\n");
    //             }
    //         }
    //     }
    // }

    // printf("dldW = %.15f\n", x_mean(linear_layer->dLdW));
    // x_print(linear_layer->dLdW);
    // printf("dldw mean = %.15f\n", x_mean(linear_layer->dLdW));
    // exit(0);
    // printf("LAYER INDEX = %d\n", linear_layer->layer_index);
    // printf("lin layer x = %.20f\n", x_mean(linear_layer->x));
    // printf("dldZ %.15f\n", x_mean(dLdz));
    
    //dLdb = mean(dLdz, axis=0, keepdims=True)
    linear_layer->dLdb = x_mean_axis(dLdz, 0);

    //dLdx = dLdz.dot(W.T)
    gsl_matrix* dLdx = x_dot(dLdz, x_transpose(linear_layer->W));
    // printf("dLdx\n");
    // x_print_sample(dLdx);

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