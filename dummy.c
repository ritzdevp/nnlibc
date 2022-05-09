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


/*
Model reference MNIST
https://colab.research.google.com/drive/1fqSAG3d0_igznPjUar3DtcQ4Ux23h0LD#scrollTo=AcslG1QeuvXv
*/
int main(){
    const gsl_rng_type * T;
    gsl_rng * rng;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    rng = gsl_rng_alloc(T);


    int train_len = 2000;
    int test_len = 100;
    
    //Note: 28x28 image is already flattened to 784 in the data
    gsl_matrix* x_train = x_scale(load_data("data/mnist_mini/x_train.dat", train_len, 784), 1.0/255.0);
    x_print_shape(x_train);
    gsl_matrix* y_train = load_data("data/mnist_mini/y_train.dat", train_len, 10); // should it be 1xtrain_len?
    x_print_shape(y_train);
    gsl_matrix* x_test = x_scale(load_data("data/mnist_mini/x_test.dat", test_len, 784), 1.0/255.0);
    x_print_shape(x_test);
    gsl_matrix* y_test = load_data("data/mnist_mini/y_test.dat", test_len, 10); //should it be 1xtest_len?
    x_print_shape(y_test);


    printf("Designing the model\n");

    // Xnet* mynet = Xnet_init(3);

    // //Hidden Layer 1
    // Linear* lin_layer1 = linear_init(784,512,0, rng);
    // xnet_add(mynet, lin_layer1);
    // Activation* act1 = Act_init("sigmoid", 1);
    // xnet_add(mynet, act1);

    // //Hidden Layer 2
    // Linear* lin_layer2 = linear_init(512,512,2, rng);
    // xnet_add(mynet, lin_layer2);
    // Activation* act2 = Act_init("sigmoid", 3);
    // xnet_add(mynet, act2);

    // //Output Layer
    // Linear* lin_layer3 = linear_init(512,10,4, rng);
    // xnet_add(mynet, lin_layer3);
    // Activation* act3 = Act_init("identity", 5);
    // xnet_add(mynet, act3);
    
    // gsl_matrix* out = net_forward(get_row(x_train, 0), mynet);
    // // x_print(out);
    // // x_print(lin_layer3->W);
    // // exit(0);;
    // // x_print(act3->y);
    // // exit(0);
    // // size_t batch_size = 1;
    // // x_print(get_row(x_train, 0));
    // // gsl_matrix* dummy = get_row(x_train, 0);

    // // x_print(y_test);
    // // for (int i = 0; i < test_len; i++){
    // //     printf("%lu\n", x_argmax_vector(get_row(y_test,i)));
    // // }
    
    // // gsl_matrix* hello = x_init(1, 5);
    // // for (int i = 0; i < 5; i++){
    // //     gsl_matrix_set(hello, 0, i, -i);
    // // }
    // // printf("THIS %lu\n", x_argmax_vector(hello));

    // // x_print(x_test);

    // int num_epochs = 3;
    // for (int epoch = 0; epoch < num_epochs; epoch++){
    //     for (int i = 0; i < 1000; i++){
    //         gsl_matrix* input = get_row(x_train, i);
    //         // printf("Dummy input %.15f\n", x_mean(input));
    //         // x_print_shape(input);
    //         gsl_matrix* output = net_forward(input, mynet);
    //         // x_print(output);
    //         // x_print_shape(output);
    //         gsl_matrix* desired = get_row(y_train, i);
    //         net_backward(desired, mynet);

    //         net_step(mynet, 0.001);
    //         net_zero_grad(mynet);
    //         Linear* lin_temp = (Linear*)mynet->layers[0];
    //         // double sum = x_mean(lin_temp->W);
    //         // printf("Sum = %.15f\n", sum);
    //     }
    //     printf("Epoch %d done.\n", epoch);
    // }
    
    // double correct_count = 0;
    // for (int i = 0; i < test_len; i++){
    //     gsl_matrix* input = get_row(x_test, i);
    //     gsl_matrix* output = net_forward(input, mynet);
    //     // x_print(output);
    //     // break;
    //     int prediction = x_argmax_vector(output);
    //     // x_print(get_row(y_test, i));
    //     int label = x_argmax_vector(get_row(y_test, i));
    //     // printf("%d %d\n", prediction, label);
    //     if (prediction == label){
    //         correct_count++;
    //     }
    // }
    // double acc = (correct_count/test_len) * 100;
    // printf("Accuracy Percentage = %f\n", acc);

    printf("Testing XOR\n");
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

    //Hidden Layer 1
    // Linear* lin_layer1 = linear_init(2,16,0, rng);
    // xnet_add(mynet, lin_layer1);
    // Activation* act1 = Act_init("relu", 1);
    // xnet_add(mynet, act1);

    // //Hidden Layer 2
    // Linear* lin_layer2 = linear_init(4,4,2, rng);
    // xnet_add(mynet, lin_layer2);
    // Activation* act2 = Act_init("relu", 3);
    // xnet_add(mynet, act2);

    // //Output Layer
    // Linear* lin_layer3 = linear_init(4,1,4, rng);
    // xnet_add(mynet, lin_layer3);
    // Activation* act3 = Act_init("sigmoid", 5);
    // xnet_add(mynet, act3);

    gsl_matrix* myinput = x_init(4,2); //row vector
    double temp_arr[4][2] = {
        {0,0}, 
        {0,1},
        {1,0},
        {1,1}
        };
    x_fill(myinput, temp_arr);
    x_print(myinput);

    gsl_matrix* labels = x_init(4,2);
    double temp_arr2[4][2] = {
        {0,1},
        {1,0},
        {1,0},
        {0,1}
    };
    x_fill(labels, temp_arr2);
    x_print(labels);

    int num_epochs = 500;
    for (int epoch = 0; epoch < num_epochs; epoch++){
        for (int i = 0; i < 4; i++){
            gsl_matrix* input = get_row(myinput, i);
            printf("input is \n");
            x_print(input);
            // printf("Dummy input %.15f\n", x_mean(input));
            // x_print_shape(input);
            gsl_matrix* output = net_forward(input, mynet);
            // printf("OUtput = \n");
            // x_print(output);
            // x_print_shape(output);
            gsl_matrix* desired = get_row(labels, i);
            net_backward(desired, mynet);

            net_step(mynet, 0.15);
            net_zero_grad(mynet);
            Linear* lin_temp = (Linear*)mynet->layers[0];
            // double sum = x_mean(lin_temp->W);
            // printf("Sum = %.15f\n", sum);
            printf("\n");
        }
        if (epoch % 100 > 0){
            printf("Epoch %d done.\n", epoch);
        }
    }

    for (int i = 0; i < 4; i++){
        gsl_matrix* input = get_row(myinput, i);
        x_print(net_forward(input, mynet));
    }


    return 0;
}