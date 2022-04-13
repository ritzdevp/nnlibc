//shape
//transpose
//element wise product
//matmul, dot product
//add
//subtract
//mean
//concat
//reshape
//sort

// void mxprint(double *arr);

#ifndef XARR
#define XARR

typedef struct Xarr {
    double ** arr;
    int row;
    int col;
} Xarr;

Xarr* Xinit(int row, int col);

void Xfree(Xarr* arr);

void x_print(Xarr* arr);

Xarr* x_add(Xarr* arr1, Xarr* arr2);

Xarr* x_sub(Xarr* arr1, Xarr* arr2);

//Element wise multiplication
Xarr* x_multiply(Xarr* arr1, Xarr* arr2);

//Multiplying a scalar to all elements
Xarr* x_scale(Xarr* arr1, double k);

Xarr* x_dot(Xarr* arr1, Xarr* arr2);

Xarr* x_transpose(Xarr* arr1);

void x_fill(Xarr* arr, void* temp);


#endif //XARR