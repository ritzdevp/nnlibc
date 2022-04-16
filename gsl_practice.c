#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include "matrix.h"


int main(){
    gsl_matrix* arr1 = x_init(3, 5);
    double w[3][5] = {
        {1,0,1,0,1},
        {0,1,0,1,0},
        {1,1,1,1,1},
    };
    x_fill(arr1, w);
    x_print(arr1);
    printf("\n");
    gsl_matrix* all_ones = x_ones(2,3);
    x_print(all_ones);
    printf("\n");
    x_print_shape(arr1);
    gsl_matrix* arr2 = x_init(3, 5);
    x_fill(arr2, w);
    gsl_matrix* sum = x_add(arr1, arr2);
    printf("\n");
    x_print(sum);
    gsl_matrix* scaledsum = x_scale(sum, -1);
    printf("\n");
    x_print(scaledsum);
    gsl_matrix* mul = x_multiply(scaledsum, arr1);
    printf("\n");
    x_print(mul);
    gsl_matrix* trans = x_transpose(mul);
    printf("\n");
    x_print(trans);
    printf("\n");
    double a[3][3] = {
        {1,2,3},
        {3,2,4},
        {10,1,0}      
    };
    gsl_matrix* atemp = x_init(3,3);
    x_fill(atemp, a);
    double b[3][4] = {
        {1,2,3,4},
        {4,5,6,7},
        {1,1,1,2}
    };
    gsl_matrix* btemp = x_init(3,4);
    x_fill(btemp, b);

    gsl_matrix* dotans = x_dot(atemp, btemp);
    printf("\n");
    x_print(dotans);
    gsl_matrix* log = x_log(dotans);
    printf("\n");
    x_print(log);
    printf("\n");
    gsl_matrix* expmat = x_exp(dotans);
    x_print(expmat);
    printf("\n");
    





}