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
    int train_len = 2000;
    int test_len = 100;
    
    //Note: 28x28 image is already flattened to 784 in the data
    gsl_matrix* x_train = x_scale(load_data("data/mnist_mini/x_train.dat", train_len, 784), 1/255);
    x_print_shape(x_train);
    gsl_matrix* y_train = load_data("data/mnist_mini/y_train.dat", train_len, 10); // should it be 1xtrain_len?
    x_print_shape(y_train);
    gsl_matrix* x_test = x_scale(load_data("data/mnist_mini/x_test.dat", test_len, 784), 1/255);
    x_print_shape(x_test);
    gsl_matrix* y_test = load_data("data/mnist_mini/y_test.dat", test_len, 10); //should it be 1xtest_len?
    x_print_shape(y_test);


    printf("Designing the model\n");

    Xnet* mynet = Xnet_init(3);

    //Hidden Layer 1
    Linear* lin_layer1 = linear_init(784,512,0);
    xnet_add(mynet, lin_layer1);
    Activation* act1 = Act_init("relu", 1);
    xnet_add(mynet, act1);

    //Hidden Layer 2
    Linear* lin_layer2 = linear_init(512,512,2);
    xnet_add(mynet, lin_layer2);
    Activation* act2 = Act_init("relu", 3);
    xnet_add(mynet, act2);

    //Output Layer
    Linear* lin_layer3 = linear_init(512,10,4);
    xnet_add(mynet, lin_layer3);
    Activation* act3 = Act_init("relu", 5);
    xnet_add(mynet, act3);
    
    // size_t batch_size = 1;
    // x_print(get_row(x_train, 0));
    // gsl_matrix* dummy = get_row(x_train, 0);

    // x_print(y_test);
    // for (int i = 0; i < test_len; i++){
    //     printf("%lu\n", x_argmax_vector(get_row(y_test,i)));
    // }
    
    // gsl_matrix* hello = x_init(1, 5);
    // for (int i = 0; i < 5; i++){
    //     gsl_matrix_set(hello, 0, i, -i);
    // }
    // printf("THIS %lu\n", x_argmax_vector(hello));

    int num_epochs = 1;
    for (int epoch = 0; epoch < num_epochs; epoch++){
        for (int i = 0; i < 1000; i++){
            gsl_matrix* input = get_row(x_train, i);
            // x_print_shape(input);
            gsl_matrix* output = net_forward(input, mynet);
            // x_print_shape(output);
            gsl_matrix* desired = get_row(y_train, i);
            net_backward(desired, mynet);

            net_step(mynet, 0.001);
            net_zero_grad(mynet);
        }
        printf("Epoch %d done.\n", epoch);
    }
    
    double correct_count = 0;
    for (int i = 0; i < test_len; i++){
        gsl_matrix* input = get_row(x_test, i);
        gsl_matrix* output = net_forward(input, mynet);
        // x_print(output);
        int prediction = x_argmax_vector(output);
        // x_print(get_row(y_test, i));
        int label = x_argmax_vector(get_row(y_test, i));
        printf("%d %d\n", prediction, label);
        if (prediction == label){
            correct_count++;
        }
    }
    double acc = (correct_count/test_len) * 100;
    printf("Accuracy Percentage = %f\n", acc);


    return 0;
}