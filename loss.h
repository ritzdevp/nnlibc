/**
 * @file loss.h
 * @brief Loss item struct and methods
 */

#ifndef LOSS
#define LOSS

#include "matrix.h"

/**
 * @brief Loss item struct
 */
typedef struct Loss_Item {
    /* Loss of shape (batch_size, 1) */
    gsl_matrix* loss;
    /* Derivative of loss wrt output y */
    gsl_matrix* loss_derivative;
} Loss_Item;

/* TODO */
gsl_matrix* mse_loss(gsl_matrix* x, gsl_matrix* y);

Loss_Item* soft_cross_ent_loss(gsl_matrix* x, gsl_matrix* y);

#endif