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
    gsl_matrix* y_train = load_data("data/mnist_mini/y_train.dat", train_len, 10);
    x_print_shape(y_train);
    gsl_matrix* x_test = x_scale(load_data("data/mnist_mini/x_test.dat", test_len, 784), 1.0/255.0);
    x_print_shape(x_test);
    gsl_matrix* y_test = load_data("data/mnist_mini/y_test.dat", test_len, 10);
    x_print_shape(y_test);


    printf("Designing the model\n");

    Xnet* mynet = Xnet_init(3);

    //Hidden Layer 1
    Linear* lin_layer1 = linear_init(784,512,0, rng);
    xnet_add(mynet, lin_layer1);
    Activation* act1 = Act_init("sigmoid", 1);
    xnet_add(mynet, act1);

    //Hidden Layer 2
    Linear* lin_layer2 = linear_init(512,512,2, rng);
    xnet_add(mynet, lin_layer2);
    Activation* act2 = Act_init("sigmoid", 3);
    xnet_add(mynet, act2);

    //Output Layer
    Linear* lin_layer3 = linear_init(512,10,4, rng);
    xnet_add(mynet, lin_layer3);
    Activation* act3 = Act_init("identity", 5);
    xnet_add(mynet, act3);
    
    int num_epochs = 3;
    for (int epoch = 0; epoch < num_epochs; epoch++){
        for (int i = 0; i < 1000; i++){
            net_zero_grad(mynet);
            gsl_matrix* input = get_row(x_train, i);
            gsl_matrix* output = net_forward(input, mynet);
            gsl_matrix* desired = get_row(y_train, i);
            net_backward(desired, mynet);
            net_step(mynet, 0.01);
        }
        printf("Epoch %d done.\n", epoch);
    }
    
    double correct_count = 0;
    for (int i = 0; i < test_len; i++){
        gsl_matrix* input = get_row(x_test, i);
        gsl_matrix* output = net_forward(input, mynet);
        int prediction = x_argmax_vector(output);
        int label = x_argmax_vector(get_row(y_test, i));
        if (prediction == label){
            correct_count++;
        }
    }
    double acc = (correct_count/(double)test_len) * 100;
    printf("Accuracy Percentage = %.3f\n", acc);

    return 0;
}