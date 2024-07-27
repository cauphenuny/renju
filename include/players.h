#pragma once

#include "board.h"

enum {
    MANUAL,
    MCTS,
};

point_t move(int, const board_t, int);
