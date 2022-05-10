#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "loss.h"

//accepts one hot vector of y
Loss_Item* soft_cross_ent_loss(gsl_matrix* x, gsl_matrix* y){
    //using logsumexp trick to avoid overflow
    //https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    
    //my logsum exp
    //https://colab.research.google.com/drive/1y0xG8OhUmGzp0-agktAMMVynlMsjZGbl#scrollTo=m5-mW1XYgfjm

    double max = gsl_matrix_get(x, 0, 0);
    for (int i = 0; i < x->size1; i++){
        for (int j = 0; j < x->size2; j++){
            if (gsl_matrix_get(x, i, j) > max){
                max = gsl_matrix_get(x, i, j);
            }
        }
    }
    gsl_matrix* all_max = x_scale(x_ones(x->size1, x->size2), max);
    
    //x - max
    gsl_matrix* x_minus_max = x_sub(x, all_max);

    gsl_matrix* exp_term = x_exp(x_minus_max);

    gsl_matrix* sum_term = x_init(exp_term->size1, 1);

    for (int i = 0; i < exp_term->size1; i++){
        double temp = 0;
        for (int j = 0; j < exp_term->size2; j++){
            temp += gsl_matrix_get(exp_term, i, j);
        }
        gsl_matrix_set(sum_term, i, 0, temp);
    }

    gsl_matrix* log_term = x_log(sum_term);
    
    gsl_matrix* log_term_temp = x_init(x_minus_max->size1, x_minus_max->size2);
    for (int i = 0; i < log_term_temp->size1; i++){
        for (int j = 0; j < log_term_temp->size2; j++){
            gsl_matrix_set(log_term_temp, i, j, gsl_matrix_get(log_term, i, 0));
        }
    }
    gsl_matrix* logsoft = x_sub(x_minus_max, log_term_temp);

    gsl_matrix* softmax = x_exp(logsoft);

    gsl_matrix* prod = x_multiply(y, logsoft);

    gsl_matrix* temp_arr = x_init(y->size1, 1);
    for (int i = 0; i < prod->size1; i++){
        double temp = 0;
        for (int j = 0; j < prod->size2; j++){
            temp += gsl_matrix_get(prod, i, j);
        }
        gsl_matrix_set(temp_arr, i, 0, temp);
    }

    gsl_matrix* loss = x_scale(temp_arr, -1);
    Loss_Item* loss_item = malloc(sizeof(Loss_Item));
    loss_item->loss = loss; //shape is (batchsize, 1)
    
    gsl_matrix* loss_deriv = x_sub(softmax, y);
    loss_item->loss_derivative = loss_deriv; //shape is (batchsize, number of classes)
    
    // x_print(loss_item->loss_derivative);
    return loss_item;
}