#pragma once

#include "board.h"

void print(const board_t);
int check(board_t b, point_t pos);
void wrap_area(board_t, int*, int*, int*, int*, int);
bool inboard(point_t);
