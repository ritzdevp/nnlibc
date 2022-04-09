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


typedef struct Xarr {
    double ** arr;
    int row;
    int col;
} Xarr;

Xarr* Xinit(int row, int col);

Xarr* x_add(Xarr* arr1, Xarr* arr2);