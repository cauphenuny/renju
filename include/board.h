#pragma once

#include <stdbool.h>

// DO NOT LET THIS GREATER THAN 16 UNLESS YOU CHANGE uint32_t TO uint64_t!
#define BOARD_SIZE 15

#define WIN_LENGTH 5

typedef struct {
    int8_t x, y;
} point_t;

typedef int board_t[BOARD_SIZE][BOARD_SIZE];

void print(const board_t);
void wrap_area(const board_t, int*, int*, int*, int*, int);
bool inboard(point_t);
char id2name(int);
void refresh(board_t board);
bool put(board_t board, int id, point_t pos);
int  check_draw(const board_t);
int  check(const board_t, point_t);
