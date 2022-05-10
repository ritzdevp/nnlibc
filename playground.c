/**
 * @file playground.c
 * @brief Main function for loading data, models, training, eval
 */

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "activations.h"
#include "layers.h"
#include "sequential.h"
#include "loss.h"
#include "data.h"

int main(){
    const gsl_rng_type * T;
    gsl_rng * rng;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    rng = gsl_rng_alloc(T);

    printf("XOR MLP\n");
    /*
        Data Label
        |0 0| 0
        |0 1| 1
        |1 0| 1
        |1 1| 0
    */

    Xnet* mynet = Xnet_init(3);

    //Hidden Layer 1
    Linear* lin_layer1 = linear_init(2,4,0, rng);
    xnet_add(mynet, lin_layer1);
    Activation* act1 = Act_init("relu", 1);
    xnet_add(mynet, act1);

    //Hidden Layer 2
    Linear* lin_layer2 = linear_init(4,4,2, rng);
    xnet_add(mynet, lin_layer2);
    Activation* act2 = Act_init("relu", 3);
    xnet_add(mynet, act2);

    //Output Layer
    Linear* lin_layer3 = linear_init(4,2,4, rng);
    xnet_add(mynet, lin_layer3);
    Activation* act3 = Act_init("sigmoid", 5);
    xnet_add(mynet, act3);

    gsl_matrix* myinput = x_init(4,2); //row vector
    double temp_arr[4][2] = {
        {0,0}, 
        {0,1},
        {1,0},
        {1,1}
        };
    x_fill(myinput, temp_arr);

    /* Labels have to be written in one-hot encoding as categorical cross entropy is being used */
    /* TODO: Binary cross entropy */
    gsl_matrix* labels = x_init(4,2);
    double temp_arr2[4][2] = {
        {0,1}, //0
        {1,0}, //1
        {1,0}, //1
        {0,1}  //0
    };
    x_fill(labels, temp_arr2);

    int num_epochs = 500;
    int train_len = 4;
    for (int epoch = 0; epoch < num_epochs; epoch++){
        double epoch_loss = 0;
        for (int i = 0; i < train_len; i++){
            gsl_matrix* input = get_row(myinput, i);
            gsl_matrix* output = net_forward(input, mynet);
            gsl_matrix* desired = get_row(labels, i);
            Loss_Item* loss_item = net_backward(desired, mynet);
            epoch_loss += gsl_matrix_get(loss_item->loss,0,0);
            net_step(mynet, 0.15);
            net_zero_grad(mynet);
        }
        if (epoch % 100 == 0){
            printf("Epoch %d Loss %.3f.\n", epoch, epoch_loss/(double)(train_len));
        }
    }
    printf("\n");

    for (int i = 0; i < 4; i++){
        gsl_matrix* input = get_row(myinput, i);
        gsl_matrix* output = net_forward(input, mynet);
        size_t arg_max = x_argmax_vector(output);
        x_print(input);
        /* NOT operator because {0,1} means 0 and {1,0} means 1 
         * So NOT of arg_max index will be output of XOR operation
        */
        printf("Out = %d\n\n", !arg_max);
    }

    return 0;
}