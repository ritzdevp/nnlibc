/**
 * @file sequential.c
 * @brief Network forward, backward, step and optimizer methods
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "sequential.h"

/**
 * @brief Initializes network
 * 
 * @param num_layers Number of layers in the network
 * Excludes input layer, includes output layer
 * @return Pointer to network
 */
Xnet* Xnet_init(int num_layers){
    Xnet* net = malloc(sizeof(Xnet));
    net->layers = malloc(sizeof(void*)*num_layers*2);
    net->layer_iterator = 0;
    /* number of layers (linear+activation) */
    net->num_layers = num_layers;
    return net;
}

/**
 * @brief Add layer to network
 * 
 * @param net Pointer to network
 * @param layer Void pointer to layer to be added
 */
void xnet_add(Xnet* net, void* layer){
    net->layers[net->layer_iterator] = layer;

    net->layer_iterator += 1;
    return;
}

/**
 * @brief Forward prop of network
 * 
 * @param input Input to the network
 * @param net Pointer to network
 * @return Pointer to forward propagation output
 */
gsl_matrix* net_forward(gsl_matrix* input, Xnet* net){
    gsl_matrix* out = forward(input, net->layers[0]);
    out = act_forward(net->layers[1], out);

    for (int i = 2; i < net->num_layers*2; i=i+2){
        /* First linear then activation, hence, linears are at even positions */
        
        /* Linear forward */
        out = forward(out, (Linear*)net->layers[i]);
        out = act_forward((Activation*)net->layers[i + 1], out);
    }

    return out;
}

/**
 * @brief Backward prop of network
 * 
 * @param target Desired output
 * @param net Pointer to network
 */
void net_backward(gsl_matrix* target, Xnet* net){
    gsl_matrix* network_output = gsl_matrix_calloc(target->size1, target->size2);

    /* Output of final activiation = output of network */
    Activation* act_final = (Activation*)(net->layers[net->num_layers*2-1]);
    gsl_matrix_memcpy(network_output, act_final->y);
    
    /* Loss item has loss and loss derivative wrt final output */
    Loss_Item* L = soft_cross_ent_loss(network_output, target);
    gsl_matrix* dLdy = x_copy(L->loss_derivative);
    
    gsl_matrix* dLdz;

    for (int i = net->num_layers*2 - 1; i >= 0; i=i-2){
        Activation* act_temp = (Activation*)(net->layers[i]);
        gsl_matrix* act_back = act_backward(act_temp);
        dLdz = x_multiply(act_back, dLdy);

        Linear* lin_temp = (Linear*)(net->layers[i - 1]);
        dLdy = backward(lin_temp, dLdz);
    }
    return;
}

/**
 * @brief Zero all gradients
 * 
 * @param net Pointer to network
 */
void net_zero_grad(Xnet* net){
    for (int i = 0; i < net->num_layers * 2; i = i + 2){
        Linear* lin_temp = (Linear*)net->layers[i];
        zerofy_matrix(lin_temp->dLdW);
        zerofy_matrix(lin_temp->dLdb);
    }
}

/**
 * @brief Update weights using stochastic gradient descent
 * 
 * @param net Pointer to network
 * @param learning_rate
 */
void net_step(Xnet* net, double learning_rate){
    for (int i = 0; i < net->num_layers * 2; i = i + 2){
        Linear* lin_temp = (Linear*)net->layers[i];

        //gsl_matrix_sub(a,b) => a=a-b
        /* 
            Didn't use x_sub because it returns a matrix
            gsl_matrix_sub subtracts from the first array itself, which is desired here
        */
        
        /* w = w - dLdz */
        /* b = b - dLdb */
        
        //No momentum
        gsl_matrix_sub(lin_temp->W, x_scale(lin_temp->dLdW, learning_rate));

        gsl_matrix_sub(lin_temp->b, x_scale(lin_temp->dLdb, learning_rate));
    }
}