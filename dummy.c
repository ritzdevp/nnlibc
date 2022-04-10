#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "matrix.h"


int main(){
    Xarr* arr1 = Xinit(2,3);
    Xarr* arr2 = Xinit(2,3);

    int count = 0;
    for (int i = 0; i < arr1->row; i++){
        for (int j = 0; j < arr1->col; j++){
            arr1->arr[i][j] = count++;
        }
    }
    for (int i = 0; i < arr2->row; i++){
        for (int j = 0; j < arr2->col; j++){
            arr2->arr[i][j] = count++;
        }
    }

    Xarr* sum = x_add(arr1, arr2);
    // x_print(sum);

    Xarr* sum2 = x_sub(arr1, arr2);
    // x_print(sum2);

    Xarr* scaled = x_scale(sum2, -1);
    x_print(scaled);
    printf("\n");
    Xarr* trans = x_transpose(scaled);
    x_print(trans);

    printf("ok \n");
    x_print(arr1);
    printf("ok \n");
    x_print(trans);
    printf("\n");
    
    
    Xarr* dotted = x_dot(arr1, trans);
    x_print(dotted);

    return 0;
}