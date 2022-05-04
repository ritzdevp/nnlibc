#ifndef ACTIV
#define ACTIV

#include "matrix.h"

typedef struct Activation {
    int layer_index; //TO be used later during network creation
    char activation[10];
    gsl_matrix* y; //output y = f(z)
    gsl_matrix* derivative; //f'(z)
    gsl_matrix* input;
    struct Activation* next;
    struct Activation* prev;
} Activation;

Activation* Act_init(char activation[10], int layer_index);

// gsl_matrix* act_identity(gsl_matrix* arr);

gsl_matrix* act_sigmoid(gsl_matrix* arr);

gsl_matrix* act_sigmoid_derivative(gsl_matrix* sigmoided_arr);

gsl_matrix* act_tanh(gsl_matrix* arr);

gsl_matrix* act_tanh_derivative(gsl_matrix* tanhed_arr);

gsl_matrix* act_relu(gsl_matrix* arr);

gsl_matrix* act_relu_derivative(gsl_matrix* relued_arr);

gsl_matrix* act_forward(Activation* activation, gsl_matrix* input);

gsl_matrix* act_backward(Activation* act);

void act_free(Activation* act);

#endif