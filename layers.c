#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "layers.h"

Linear* linear_init(int input_size, int output_size, int layer_index){
    Linear* linear_layer = malloc(sizeof(Linear));

    linear_layer->input_size = input_size;
    linear_layer->output_size = output_size;
    linear_layer->layer_index = layer_index;

    linear_layer->W = Xinit(input_size, output_size);
    linear_layer->b = Xinit(1, output_size);
    
    return linear_layer;
}

Xarr* forward(Xarr* input, Linear* linear_layer){
    //z = x.dot(W) + b
    linear_layer->z = x_add(x_dot(input, linear_layer->W), linear_layer->b);
    return linear_layer->z;
}

void linear_free(Linear* linear_layer){
    Xfree(linear_layer->W);
    Xfree(linear_layer->b);
    //try except
    if (linear_layer->z != NULL){
        Xfree(linear_layer->z);
    }
    free(linear_layer);
    return;
}

