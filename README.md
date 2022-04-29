# Neural Network in C + WASM Compilation

## Overview

This is a Pytorch-like neural network library written in C.

## Usage

#### 1. Activations


#### 2. Layers

#### 3. Examples


## GNU Scientific Library Wrapper

The GNU Scientific Library is used for matrix and vector operations and a matrix.c wrapper can be invoked to perform those operations. 

#### Compiling GSL to C
Please follow the steps given [HERE](https://coral.ise.lehigh.edu/jild13/2016/07/11/hello/).
#### Compiling GSL to WASM
Install [Emscripten](https://emscripten.org/docs/getting_started/downloads.html).

Download the 'Current Stable Version' from the [GSL webpage](https://www.gnu.org/software/gsl/).
In the downloaded GSL folder, say gsl-2.7.1, run the following commands

```
emconfigure ./configure
emmake make LDFLAGS=-all-static
sudo make install
```

GSL will be installed at /opt/gsl-2.7.1 with WASM executables.





