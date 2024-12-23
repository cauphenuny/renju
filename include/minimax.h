#ifndef MINIMAX_H
#define MINIMAX_H

#include "board.h"
#include "game.h"

typedef struct {
    bool parallel;
    int max_depth;
    struct {
        bool begin_vct, look_forward;
        int leaf_vct_depth;
    } optim;
    struct {
        int adjacent;
    } strategy;
} minimax_param_t;

point_t minimax(game_t game, const void* assets);

#endif