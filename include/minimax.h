#ifndef MINIMAX_H
#define MINIMAX_H

#include "board.h"
#include "game.h"

typedef struct {
    bool use_vct;
    int max_depth;
} minimax_param_t;

point_t minimax(game_t game, const void* assets);

#endif