#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "activations.h"
#include "layers.h"
#include "sequential.h"
#include "loss.h"


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
    gsl_matrix* xinput = x_init(1,3);
    gsl_matrix_set(xinput, 0, 0, 1);
    gsl_matrix_set(xinput, 0,0, 1);
    gsl_matrix_set(xinput, 0,1,2);
    gsl_matrix_set(xinput, 0,2,3);
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

    gsl_matrix* z = forward(xinput, lin);
    // linear_free(lin);
    x_print(z);
    printf("\n");
    gsl_matrix* y = act_identity(z);
    Linear* lin2 = linear_init(5,2,1);
    x_print(forward(y, lin2));
    x_print(y);
    printf("\n");

    Xnet* net = Xnet_init(2);
    net->layers[0] = lin;

    Activation* act = Act_init("tanh", 1);
    net->layers[1] = act;
    x_print(act_forward(net->layers[1], z));

    printf("%d\n", net->layers[0] == lin);
    printf("%d\n", net->layers[1] == lin2);
    
    double x[2][3] = {
        {1,2,3},
        {4,5,6}
    };
    double yy[2][3] = {
        {1,1,2},
        {2,1,3}
    };

    gsl_matrix* X = x_init(2,3);
    gsl_matrix* Y = x_init(2,3);
    x_fill(X, x);
    x_fill(Y, yy);
    printf("OK! \n");
    x_print(soft_cross_ent_loss(X, Y));

    printf("=====\nNEURAL NETWORK\n");
    gsl_matrix* myxinput = x_init(1,2); //row vector

    //Making model
    Linear* layer1 = linear_init(2,3,1); //layer 1, layer 0 is input itself
    Activation* act1 = Act_init("relu", 2);
    Linear* layer2 = linear_init(3,2,3);
    Activation* act2 = Act_init("sigmoid", 4);

    //Manually setting weights
    double w1[2][3] = {
        {1,2,3},
        {3,2,1}
    };
    double b1[1][3] = {{1,0,1}};
    x_fill(layer1->W, w1);
    x_fill(layer1->b, b1);

    double w2[3][2] = {
        {1,0},
        {2,0},
        {1,1}
    };
    double b2[1][3] = {{1,2}};
    x_fill(layer2->W, w2);
    x_fill(layer2->b, b2);

    //forward
    gsl_matrix* out = forward(myxinput, layer1);
    out = act_forward(act1, out);
    out = forward(out, layer2);
    out = act_forward(act2, out);


    gsl_matrix* desired = x_init(1, 2);
    double temp[1][2] = {{1,0}};
    x_fill(desired, temp);

    //finding softmax crossentropy loss
    x_print(soft_cross_ent_loss(out, desired));

    return 0;
}