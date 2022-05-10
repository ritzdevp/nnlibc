/**
 * @file activations.h
 * @brief Activation struct and methods
 */

#ifndef ACTIV
#define ACTIV

#include "matrix.h"

/**
 * @brief Activation struct
 */
typedef struct Activation {

    /* Index of this layer in the network */
    int layer_index;
    /* String specifying type of activation; relu, sigmoid, identity, tanh */
    char activation[10];
    /* Output of activation layer y=f(z) */
    gsl_matrix* y;
    /* Derivative of activation */
    gsl_matrix* derivative; //f'(z)
    /* Input to the activation layer */
    gsl_matrix* input;
    /* Pointer to next activation layer [NOT used in the design yet] */
    struct Activation* next;
    /* Pointer to previous activation layer [NOT used in the design yet] */
    struct Activation* prev;
} Activation;

Activation* Act_init(char activation[10], int layer_index);

gsl_matrix* act_identity(gsl_matrix* arr);

gsl_matrix* act_identity_derivative(gsl_matrix* arr);

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