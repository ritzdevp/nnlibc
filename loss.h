#ifndef LOSS
#define LOSS

//mean squared error loss
//cross entropy loss

#include "matrix.h"

typedef struct Loss_Item {
    gsl_matrix* loss;
    gsl_matrix* loss_derivative; //wrt y
} Loss_Item;

gsl_matrix* mse_loss(gsl_matrix* x, gsl_matrix* y);

//x are logits (network output, without softmax) and y are labels
Loss_Item* soft_cross_ent_loss(gsl_matrix* x, gsl_matrix* y);

#endif