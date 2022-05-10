/**
 * @file data.c
 * @brief Methods for loading data 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "data.h"

/**
 * @brief Load .dat binary file into gsl matrix
 * 
 * @param data_path Path for the binary file
 * @param rows Number of rows in resulting matrix
 * @param cols Number of cols in resulting matrix
 * @return Pointer to matrix
 */
gsl_matrix* load_data(char* data_path, size_t rows, size_t cols){
    gsl_matrix* data = x_init(rows,cols);

    /* read binary file */
    FILE * f = fopen (data_path, "rb");
    gsl_matrix_fread(f, data);
    fclose (f);
    return data;
}