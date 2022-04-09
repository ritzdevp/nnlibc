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
    for (int i = 0; i < sum->row; i++){
        for (int j = 0; j < sum->col; j++){
            printf("%f ",sum->arr[i][j]);
        }
        printf("\n");
    }
    return 0;
}