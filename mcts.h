#pragma once

#include "board.h"

typedef struct {
    double C;
    int M;
} mcts_parm_t; // parameters for mcts

point_t mcts(const board_t, int, const mcts_parm_t);
