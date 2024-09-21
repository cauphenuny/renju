#ifndef PLAYERS_H
#define PLAYERS_H

#include "board.h"
#include "game.h"

enum {
    MANUAL,
    MCTS,
    MCTS2,
    MINIMAX,
    MIX,
};

point_t move(int, const game_t);

void players_init();

#endif