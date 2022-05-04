#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "data.h"

gsl_matrix* load_data(char* data_path, size_t rows, size_t cols){
    gsl_matrix* data = x_init(rows,cols);
    FILE * f = fopen (data_path, "rb"); //rb means read binary file
    gsl_matrix_fread(f, data);
    fclose (f);
    return data;
}