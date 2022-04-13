//Linear layer

#include "matrix.h"

typedef struct Linear {
    int input_size;
    int output_size;
    int layer_index; //TO be used later during network creation
    Xarr* W; //Weight matrix
    Xarr* b; //bias vector
    Xarr* z; //output vector
} Linear;

Linear* linear_init(int input_size, int output_size, int layer_index);

void linear_free(Linear* linear_layer);

Xarr* forward(Xarr* input, Linear* linear_layer);