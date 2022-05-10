/**
 * @file sequential.h
 * @brief Network struct and methods
 */

#include "matrix.h"
#include "activations.h"
#include "layers.h"
#include "loss.h"

/**
 * @brief Network struct
 */
typedef struct Xnet {
    /* Number of layers in the network excluding input layer and including output layer */
    int num_layers;
    
    /* Tracks forward/backward propagation along network */
    int layer_iterator;
    
    /* pointer of pointers pointing to layers */
    void** layers;
} Xnet;

Xnet* Xnet_init(int num_layers);

void xnet_add(Xnet* net, void* layer);

gsl_matrix* net_forward(gsl_matrix* input, Xnet* net);

Loss_Item* net_backward(gsl_matrix* target, Xnet* net);

void net_zero_grad(Xnet* net);

void net_step(Xnet* net, double learning_rate);