#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "sequential.h"

Xnet* Xnet_init(int num_layers){
    Xnet* net = malloc(sizeof(Xnet));
    net->layers = malloc(sizeof(void*)*num_layers);
}

void xnet_add(Xnet* net, void* layer, int layer_index){
    net->layers[layer_index] = layer;
    return;
}