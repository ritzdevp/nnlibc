/**
 * @file matrix.h
 * @brief Numpylike wrapper over gsl methods
 */

#ifndef XARR
#define XARR

#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_statistics.h>
 #include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

gsl_matrix* x_init(int row, int cols);

gsl_matrix* x_ones(int row, int col);

void zerofy_matrix(gsl_matrix* arr);

void x_free(gsl_matrix* arr);

void x_print(gsl_matrix* arr);

void x_print_shape(gsl_matrix* arr);

gsl_matrix* x_add(gsl_matrix* arr1, gsl_matrix* arr2);

gsl_matrix* x_sub(gsl_matrix* arr1, gsl_matrix* arr2);

gsl_matrix* x_multiply(gsl_matrix* arr1, gsl_matrix* arr2);

gsl_matrix* x_scale(gsl_matrix* arr1, double k);

gsl_matrix* x_log(gsl_matrix* arr);

gsl_matrix* x_exp(gsl_matrix* arr);

gsl_matrix* x_dot(gsl_matrix* arr1, gsl_matrix* arr2);

gsl_matrix* x_transpose(gsl_matrix* arr1);

double x_mean(gsl_matrix* arr);

gsl_matrix* x_mean_axis(gsl_matrix* arr, int axis);

void x_fill(gsl_matrix* arr, void* temp);

gsl_matrix* x_copy(gsl_matrix* src);

gsl_matrix* x_broadcast_vector(gsl_matrix* arr, int rows);

gsl_matrix* get_row(gsl_matrix* arr, size_t row_index);

size_t x_argmax_vector(gsl_matrix* arr);

void x_xavier_init(gsl_matrix* arr, gsl_rng * rng);

void x_print_sample(gsl_matrix* arr);

#endif //XARR