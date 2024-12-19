#ifndef MINIMAX_H
#define MINIMAX_H

#include "board.h"
#include "game.h"

typedef struct {
    int max_depth;
    bool use_vct, use_parallel;
} minimax_param_t;

point_t minimax(game_t game, const void* assets);

#endif