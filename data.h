#ifndef DATA
#define DATA

#include "matrix.h"

//data should be in numpy format
gsl_matrix* load_data(char* data_path, size_t rows, size_t cols);

#endif