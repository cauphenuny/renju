#ifndef MINIMAX_H
#define MINIMAX_H

#include "board.h"
#include "game.h"

point_t minimax(const game_t game);
int ab_evaluate(board_t board, point_t pos, int sgn);

#endif