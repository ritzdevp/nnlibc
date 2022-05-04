
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

  
  

### Compiling the Neural Network Library in C

`gcc -Wall *.c -lm -lgsl -lgslcblas -o Output`

### Compiling the Neural Network Library to WASM

`emcc *.c -o Output.wasm -I/opt/gsl-2.7.1/include -L/opt/gsl-2.7.1/lib -lgsl -lm -lgslcblas -lc -s STANDALONE_WASM`

  

### Running WASM files

  

#### Using Wasmer

Install [Wasmer](https://github.com/wasmerio/wasmer) using the following command.

  

`curl https://get.wasmer.io -sSfL | sh`

  

Run Wasm output file

  

`wasmer Output.wasm`

  

#### Using SilverLine Linux Runtime

  

Follow the Setup steps [here](https://github.com/SilverLineFramework/silverline/wiki/Local-Testing-Guide).

  

Running the Wasm output file,

  

Open 4 terminal windows.

  

Terminal 0: MQTT

  

`mosquitto`

  

Terminal 1: Orchestrator

  

```

cd orchestrator/arts-main

make run

```

  

Terminal 2: Linux Runtime

  

In this example, 'dir' is at '/home/ritzdevp/nnlibc'; replace it with the path of the neural network library.

```

./runtime-linux/runtime --host=localhost:1883 --name=test --dir=/home/ritzdevp/nnlibc --appdir=/home/ritzdevp/nnlibc

```

  

Terminal3: Run

  

`python3 libsilverline/run.py --path Output.wasm --runtime test`

  

The output will be visible in Terminal 2.

# Using the Neural Network Library
## Loading Data
This library reads the training and testing data from .dat files. The np2dat.py file can be used to convert a numpy file to a .dat file which can be read into a gsl_matrix* by calling the load_data() function.
Here is an example,
Consider that you have a numpy file in data/mnist_mini/x_train.npy.
Convert this numpy file to .dat using the following command

    python3 np2dat.py data/mnist_mini/x_train.npy
The file data/mnist_mini/x_train.dat should be visible now.
The data can be loaded into a gsl_matrix as shown below.

```
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "data.h"
int main(){
int train_len = 2000;
	//Note: 28x28 image is already flattened to 784 in the data
	gsl_matrix* x_train = load_data("data/mnist_mini/x_train.dat", train_len, 784);
	x_print_shape(x_train);
	return  0;
}
```
The output will be 

    shape = (2000, 784)
