/**
 * @file data.h
 * @brief Data methods
 */

#ifndef DATA
#define DATA

#include "matrix.h"

gsl_matrix* load_data(char* data_path, size_t rows, size_t cols);

#endif