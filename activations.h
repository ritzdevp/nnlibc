#include "matrix.h"

typedef struct Activation {
    int layer_index; //TO be used later during network creation
    char activation[10];
    Xarr* y;
    struct Activation* next;
    struct Activation* prev;
} Activation;

Xarr* act_identity(Xarr* arr);

Xarr* act_sigmoid(Xarr* arr);

Xarr* act_tanh(Xarr* arr);

Xarr* act_relu(Xarr* arr);

Activation* Act_init(char activation[10], int layer_index);

Xarr* act_forward(Activation* activation, Xarr* input);

void act_free(Activation* act);