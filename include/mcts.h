#ifndef MCTS_H
#define MCTS_H

#include "board.h"
#include "game.h"
#include "neuro.h"
#include "zobrist.h"

typedef struct {
    double C, start_c, end_c;
    int min_time;
    int max_time;
    int min_count;
    int8_t wrap_rad;
} mcts_parm_t; // parameters for mcts

point_t mcts(game_t, mcts_parm_t);

#endif