#ifndef TRIVIAL_H
#define TRIVIAL_H

#include "game.h"
#include "board.h"

point_t random_move(game_t game);

point_t trivial_move(board_t board, int cur_id, double time_limit, bool use_vct);

#endif