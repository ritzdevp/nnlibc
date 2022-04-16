//mean squared error loss
//cross entropy loss

#include "matrix.h"

gsl_matrix* mse_loss(gsl_matrix* x, gsl_matrix* y);

//x are logits and y are labels
gsl_matrix* soft_cross_ent_loss(gsl_matrix* x, gsl_matrix* y);