#pragma once

#include "board.h"
#include "game.h"

enum {
    MANUAL,
    MCTS,
    MCTS2,
};

point_t move(int, const game_t);

void players_init();
