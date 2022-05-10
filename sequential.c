#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "sequential.h"

//Does not include input layers
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
    gsl_matrix* out = forward(input, net->layers[0]);
    out = act_forward(net->layers[1], out);
    // printf("Input %.15f\n", x_mean(input));
    
    // printf("Mean = %.15f\n", x_mean(out));
    for (int i = 2; i < net->num_layers*2; i=i+2){
        //First linear then activation, hence, linears are at even positions
        out = forward(out, (Linear*)net->layers[i]); //linear forward
        out = act_forward((Activation*)net->layers[i + 1], out);
        // printf("Mean = %.15f\n", x_mean(out));
    }

    return out;
}

//target is the desired output
void net_backward(gsl_matrix* target, Xnet* net){
    gsl_matrix* network_output = gsl_matrix_calloc(target->size1, target->size2);

    //output of final activiation = output of network
    Activation* act_final = (Activation*)(net->layers[net->num_layers*2-1]);
    gsl_matrix_memcpy(network_output, act_final->y);
    // printf("Net out\n");
    // x_print_sample(network_output);

    
    //loss item has loss and loss derivative wrt final output
    Loss_Item* L = soft_cross_ent_loss(network_output, target);
    // printf("Loss\n");
    // x_print(L->loss);
    // printf("Loss Deriv\n");
    // x_print_sample(L->loss_derivative);

    // printf("Loss derivative %.15f\n", x_mean(L->loss_derivative));
    // x_print(L->loss_derivative);
    gsl_matrix* dLdy = x_copy(L->loss_derivative);
    // printf("Dldy\n");
    // x_print_sample(dLdy);
    
    gsl_matrix* dLdz;

    for (int i = net->num_layers*2 - 1; i >= 0; i=i-2){
        //last layer is activation, second last is linear and so on

        Activation* act_temp = (Activation*)(net->layers[i]);
        gsl_matrix* act_back = act_backward(act_temp);
        // printf("Act back\n");
        // x_print_sample(act_back);
        dLdz = x_multiply(act_back, dLdy);
        // printf("dLdZ\n");
        // x_print_sample(dLdz);

        if (i == 5){
            // printf("Act back\n");
            // x_print_shape(act_backward(act_temp));
            // x_print(act_backward(act_temp));
            // printf("dldZ\n");
            // x_print(dLdz);
        }
        // printf("HERE LAYER = %d\n", i);
        // printf("%.15f\n", x_mean(dLdy));
        // printf("%.15f\n", x_mean(dLdz));
        // printf("Dldy %.15f\n", x_mean(dLdy));
        // printf("Act %.15f\n", x_mean(act_backward(act_temp)));

        Linear* lin_temp = (Linear*)(net->layers[i - 1]);
        dLdy = backward(lin_temp, dLdz);
        // printf("dLdy for layer %d\n", i - 1);
        // x_print_sample(dLdy);
    }
    return;
}

void net_zero_grad(Xnet* net){
    for (int i = 0; i < net->num_layers * 2; i = i + 2){
        Linear* lin_temp = (Linear*)net->layers[i];
        // printf("initial\n");
        // x_print_sample(lin_temp->dLdW);
        zerofy_matrix(lin_temp->dLdW);
        // printf("after\n");
        // x_print_sample(lin_temp->dLdW);
        // exit(0);
        zerofy_matrix(lin_temp->dLdb);
    }
}

void net_step(Xnet* net, double learning_rate){
    for (int i = 0; i < net->num_layers * 2; i = i + 2){
        Linear* lin_temp = (Linear*)net->layers[i];

        //gsl_matrix_sub(a,b) => a=a-b
        //Didn't use x_sub because it returns a matrix
        //gsl_matrix_sub subtracts from the first array itself, which is desired here
        //w = w - dLdz
        //b = b - dLdb
        
        //No momentum
        // printf("meanW = %.15f\n", x_mean(lin_temp->W));
        // if (i % 2 == 0){
        //     printf("Layer = %d\n", i);
        //     printf("meandLdW = %.15f\n", x_mean(lin_temp->dLdW));
        // }

        // x_print(lin_temp->W);
        // printf("Layer = %d\n", i);
        // printf("dLdW for update\n");
        // x_print_sample(lin_temp->dLdW);
        // printf("W update initial \n");
        // x_print_sample(lin_temp->W);
        gsl_matrix_sub(lin_temp->W, x_scale(lin_temp->dLdW, learning_rate));
        // printf("W update after \n");
        // x_print_sample(lin_temp->W);
        // printf("\n");

        // gsl_matrix* temp = get_row(lin_temp->dLdW, 0);
        // printf("Layer = %d\n", i);
        // for (int i = 0; i < 5; i++){
        //     printf("%.4f ", gsl_matrix_get(temp, 0, i));
        // }
        // printf("\n\n");
        // printf("OK\n");
        // x_print(lin_temp->W);
        gsl_matrix_sub(lin_temp->b, x_scale(lin_temp->dLdb, learning_rate));
    }
}