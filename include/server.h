#pragma once

#include "board.h"

void print(const board_t);
void wrap_area(board_t, int*, int*, int*, int*, int);
bool inboard(point_t);
int check(board_t, point_t);
int game(int player1, int player2, int first);
