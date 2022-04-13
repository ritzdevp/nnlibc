#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "activations.h"
#include "layers.h"
#include "sequential.h"


int main(){
    // Xarr* arr1 = Xinit(2,3);
    // Xarr* arr2 = Xinit(2,3);

    // int count = 0;
    // for (int i = 0; i < arr1->row; i++){
    //     for (int j = 0; j < arr1->col; j++){
    //         arr1->arr[i][j] = count++;
    //     }
    // }

    // printf("arr1\n");
    // x_print(arr1);
    // printf("\n");


    // for (int i = 0; i < arr2->row; i++){
    //     for (int j = 0; j < arr2->col; j++){
    //         arr2->arr[i][j] = count++;
    //     }
    // }

    // Xarr* sum = x_add(arr1, arr2);
    // // x_print(sum);

    // Xarr* sum2 = x_sub(arr1, arr2);
    // // x_print(sum2);

    // Xarr* scaled = x_scale(sum2, -1);
    // x_print(scaled);
    // printf("\n");
    // Xarr* trans = x_transpose(scaled);
    // x_print(trans);

    // printf("ok \n");
    // x_print(arr1);
    // printf("ok \n");
    // x_print(trans);
    // printf("\n");
    
    
    // Xarr* dotted = x_dot(arr1, trans);
    // x_print(dotted);

    // Xarr* iden = act_identity(dotted);
    // printf("\n");
    // x_print(iden);

    // Xarr* sig = act_sigmoid(arr1);
    // printf("\n");
    // x_print(sig);

    // Xarr* tan = act_tanh(sig);
    // printf("\n");
    // x_print(tan);

    // tan->arr[0][0] = -1.234;
    // printf("\n");
    // x_print(act_relu(tan));


    // TRYING LINEAR LAYER
    Xarr* xinput = Xinit(1,3);
    xinput->arr[0][0] = 1;
    xinput->arr[0][1] = 2;
    xinput->arr[0][2] = 3;
    x_print(xinput);
    printf("\n");

    Linear* lin = linear_init(3, 5, 0);
    double w[3][5] = {
        {1,0,1,0,1},
        {0,1,0,1,0},
        {1,1,1,1,1},
    };
    double b[1][5] = {{-3,-4,-5,-6,-7}};

    x_fill(lin->W, w);
    x_fill(lin->b, b);
    // printf("%f\n", *((w+2)[0]));

    Xarr* z = forward(xinput, lin);
    // linear_free(lin);
    x_print(z);
    printf("\n");
    Xarr* y = act_identity(z);
    Linear* lin2 = linear_init(5,2,1);
    x_print(forward(y, lin2));
    // x_print(y);
    printf("\n");

    Xnet* net = Xnet_init(2);
    net->layers[0] = lin;

    Activation* act = Act_init("tanh", 1);
    net->layers[1] = act;
    x_print(act_forward(net->layers[1], z));

    printf("%d\n", net->layers[0] == lin);
    printf("%d\n", net->layers[1] == lin2);
    

    

    return 0;
}