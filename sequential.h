#include "matrix.h"

typedef struct Xnet {
    int num_layers; //excluding input layer and including output layer
    //note: activation layer is also considered a layer here
    void** layers;
} Xnet;

Xnet* Xnet_init(int num_layers);

void xnet_add(Xnet* net, void* layer, int layer_index);