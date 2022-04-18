#include "matrix.h"
#include "activations.h"
#include "layers.h"
#include "loss.h"

typedef struct Xnet {
    int num_layers; //excluding input layer and including output layer
    //note: activation layer is also considered a layer here
    
    int layer_iterator;
    
    //pointer of pointers pointing to layers
    void** layers;
} Xnet;

Xnet* Xnet_init(int num_layers);

void xnet_add(Xnet* net, void* layer);

gsl_matrix* net_forward(gsl_matrix* input, Xnet* net);

void net_backward(gsl_matrix* target, Xnet* net);

void zero_grad(Xnet* net);