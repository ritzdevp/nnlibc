#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "sequential.h"

Xnet* Xnet_init(int num_layers){
    Xnet* net = malloc(sizeof(Xnet));
    net->layers = malloc(sizeof(void*)*num_layers*2); //factor 2 because of activation layer
    net->layer_iterator = 0;
    net->num_layers = num_layers; //number of dense layers (linear+activation)
    return net;
}

void xnet_add(Xnet* net, void* layer){
    net->layers[net->layer_iterator] = layer;

    net->layer_iterator += 1;
    return;
}

gsl_matrix* net_forward(gsl_matrix* input, Xnet* net){
    // printf("IN net forward\n");
    gsl_matrix* out = forward(input, net->layers[0]);
    // printf("first forward\n");
    out = act_forward(net->layers[1], out);
    for (int i = 2; i < net->num_layers*2; i=i+2){
        //First linear then activation, hence, linears are at even positions
        out = forward(out, (Linear*)net->layers[i]); //linear forward
        // printf("temp forward\n");
        out = act_forward((Activation*)net->layers[i + 1], out);
        // printf("temp act\n");
    }

    return out;
}

void net_backward(gsl_matrix* target, Xnet* net){
    gsl_matrix* network_output = gsl_matrix_calloc(target->size1, target->size2);

    //output of final activiation = output of network
    Activation* act_final = (Activation*)(net->layers[net->num_layers*2-1]);
    gsl_matrix_memcpy(network_output, act_final->y);
    
    //loss item has loss and loss derivative wrt final output
    Loss_Item* L = soft_cross_ent_loss(network_output, target);

    gsl_matrix* dLdy = x_copy(L->loss_derivative);

    gsl_matrix* dLdz;

    for (int i = net->num_layers*2 - 1; i >= 0; i=i-2){
        //last layer is activation, second last is linear and so on

        Activation* act_temp = (Activation*)(net->layers[i]);
        dLdz = x_multiply(act_backward(act_temp, act_temp->input), dLdy);
        Linear* lin_temp = (Linear*)(net->layers[i - 1]);
        dLdy = backward(lin_temp, dLdz);
    }
    return;
}

void zero_grad(Xnet* net){
    for (int i = 0; i < net->num_layers * 2; i = i + 2){
        Linear* lin_temp = (Linear*)net->layers[i];
        zerofy_matrix(lin_temp->dLdW);
        zerofy_matrix(lin_temp->dLdb);
    }
}

void step(double learning_rate){
    for (int i = 0; i < net->num_layers * 2; i = i + 2){
        Linear* lin_temp = (Linear*)net->layers[i];

        //gsl_matrix_sub(a,b) => a=a-b
        //Didn't use x_sub because it returns a matrix
        //gsl_matrix_sub subtracts from the first array itself, which is desired here
        //w = w - dLdz
        //b = b - dLdb
        gsl_matrix_sub(lin_temp->W, x_scale(lin_temp->dLdW, learning_rate));
        gsl_matrix_sub(lin_temp->b, x_scale(lin_temp->dLdb, learning_rate));
    }
}