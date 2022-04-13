#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "activations.h"
#include "layers.h"


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
    double b[5] = {-3,-4,-5,-6,-7};
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 5; j++){
            lin->W->arr[i][j] = w[i][j];
        }
    }
    for (int i = 0; i < 5; i++){
        lin->b->arr[0][i] = b[i];
    }


    Xarr* z = forward(xinput, lin);
    x_print(z);

    return 0;
}