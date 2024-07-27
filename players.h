#pragma once

enum {
    MANUAL,
    MCTS,
};

point_t move(int, const board_t, int);
