#pragma once
#define BOARD_SIZE 11
#define WIN_LENGTH 5

typedef struct {
    int x, y;
} point_t;

typedef int board_t[BOARD_SIZE][BOARD_SIZE];
typedef int (*board_p)[BOARD_SIZE];
