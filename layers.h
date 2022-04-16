//Linear layer

#include "matrix.h"

typedef struct Linear {
    int input_size;
    int output_size;
    int layer_index; //TO be used later during network creation
    char activation[10];
    gsl_matrix* W; //Weight matrix
    gsl_matrix* b; //bias vector
    gsl_matrix* z; //output vector
    struct Linear* next;
    struct Linear* prev;
} Linear;

Linear* linear_init(int input_size, int output_size, int layer_index);

void linear_free(Linear* linear_layer);

gsl_matrix* forward(gsl_matrix* input, Linear* linear_layer);