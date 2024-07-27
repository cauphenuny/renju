#pragma once

#include "board.h"

enum {
    MANUAL,
    MCTS,
    MCTS2,
};

point_t move(int, const board_t, int);
