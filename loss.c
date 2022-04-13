#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "loss.h"

Xarr* soft_cross_ent_loss(Xarr* x, Xarr* y){
    //using logsumexp trick to avoid overflow
    //https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    
    //my logsum exp
    //https://colab.research.google.com/drive/1y0xG8OhUmGzp0-agktAMMVynlMsjZGbl#scrollTo=m5-mW1XYgfjm

    double max = x->arr[0][0];
    for (int i = 0; i < x->row; i++){
        for (int j = 0; j < x->col; j++){
            if (x->arr[i][j] > max){
                max = x->arr[i][j];
            }
        }
    }
    Xarr* all_max = x_scale(x_ones(x->row, x->col), max);
    
    //x - max
    Xarr* x_minus_max = x_sub(x, all_max);
    // printf("XMINUSMAXNUM\n");
    // x_print(x_minus_max);
    // printf("\n");

    Xarr* exp_term = x_exp(x_minus_max);
    // printf("EXP TERM\n");
    // x_print(exp_term);
    // printf("\n");

    Xarr* sum_term = Xinit(exp_term->row, 1);

    for (int i = 0; i < exp_term->row; i++){
        double temp = 0;
        for (int j = 0; j < exp_term->col; j++){
            temp += exp_term->arr[i][j];
        }
        sum_term->arr[i][0] = temp;
    }

    // printf("SUM TERM\n");
    x_print(sum_term);
    // printf("\n");

    // printf("LOG TERM\n");
    Xarr* log_term = x_log(sum_term);
    // x_print(log_term);
    // printf("\n");
    

    // printf("LOGSOFT\n");
    Xarr* log_term_temp = Xinit(x_minus_max->row, x_minus_max->col);
    for (int i = 0; i < log_term_temp->row; i++){
        for (int j = 0; j < log_term_temp->col; j++){
            log_term_temp->arr[i][j] = log_term->arr[i][0];
        }
    }
    Xarr* logsoft = x_sub(x_minus_max, log_term_temp);
    // x_print(logsoft);
    // printf("\n");

    // printf("SOFTMAX\n");
    Xarr* softmax = x_exp(logsoft);
    // x_print(softmax);
    // printf("\n");

    // printf("PROD\n");
    Xarr* prod = x_multiply(y, logsoft);
    // x_print(prod);
    // printf("\n");

    Xarr* temp_arr = Xinit(y->row, 1);
    for (int i = 0; i < prod->row; i++){
        double temp = 0;
        for (int j = 0; j < prod->col; j++){
            temp += prod->arr[i][j];
        }
        temp_arr->arr[i][0] = temp;
    }

    Xarr* loss = x_scale(temp_arr, -1);
    // printf("LOSS\n");
    // x_print(loss);
    // printf("\n");
    return loss;
}