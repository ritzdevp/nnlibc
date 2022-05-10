/**
 * @file layers.h
 * @brief Linear layer struct and methods
 */

#ifndef LINEAR
#define LINEAR

#include "matrix.h"

/**
 * @brief Linear layer struct
 */
typedef struct Linear {
    /* Input dimension of linear layer */
    int input_size;
    /* Output dimension of linear layer */
    int output_size;
    /* Index of this layer in the network */
    int layer_index;

    /* Input to this layer */
    gsl_matrix* x;
    /* Weight matrix */
    gsl_matrix* W;
    /* Bias */
    gsl_matrix* b;
    /* Output of linear layer */
    gsl_matrix* z;

    /* Gradient of loss wrt layer weights */
    gsl_matrix* dLdW;
    /* Gradient of loss wrt biases */
    gsl_matrix* dLdb;

    /* Pointer to next layer [NOT used in the design yet] */
    struct Linear* next;
    /* Pointer to previous layer [NOT used in the design yet] */
    struct Linear* prev;
    
    /* Random number generator */
    gsl_rng * rng;
} Linear;

Linear* linear_init(int input_size, int output_size, int layer_index, gsl_rng * rng);

void linear_free(Linear* linear_layer);

gsl_matrix* forward(gsl_matrix* input, Linear* linear_layer);

gsl_matrix* backward(Linear* linear_layer, gsl_matrix* dLdz);

#endif