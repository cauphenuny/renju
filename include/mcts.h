#ifndef MCTS_H
#define MCTS_H

#include "board.h"
#include "game.h"
#include "neuro.h"
#include "zobrist.h"

typedef struct {
    double C, start_c, end_c;
    int min_time;
    int min_count;
    int8_t wrap_rad;
    bool check_forbid;
    bool dynamic_area;
} mcts_param_t;  // parameters for mcts

point_t mcts(game_t, void*);

#endif