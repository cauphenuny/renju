#pragma once

#include "board.h"

typedef struct {
    double C;
    int TIME_LIMIT;
    int MIN_COUNT;
    int WRAP_RAD;
    int M;
} mcts_parm_t; // parameters for mcts

point_t mcts(const board_t, int, const mcts_parm_t);
