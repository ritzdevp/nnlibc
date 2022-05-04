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
    int train_len = 2000;
    int test_len = 100;
    
    //Note: 28x28 image is already flattened to 784 in the data
    gsl_matrix* x_train = load_data("data/mnist_mini/x_train.dat", train_len, 784);
    x_print_shape(x_train);
    gsl_matrix* y_train = load_data("data/mnist_mini/y_train.dat", train_len, 10); // should it be 1xtrain_len?
    x_print_shape(y_train);
    gsl_matrix* x_test = load_data("data/mnist_mini/x_test.dat", test_len, 784);
    x_print_shape(x_test);
    gsl_matrix* y_test = load_data("data/mnist_mini/y_test.dat", test_len, 10); //should it be 1xtest_len?
    x_print_shape(y_test);



    
    

    return 0;
}