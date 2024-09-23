#ifndef PLAYERS_H
#define PLAYERS_H

#include "board.h"
#include "game.h"
#include "mcts.h"

enum {
    MANUAL,
    MCTS,     // default MCTS
    MCTS2,    // for test
    MCTS_NN,  // MCTS with neural network
    MINIMAX,
    MIX,
};

extern mcts_parm_t mcts_preset;

point_t move(int, void* player_assets, const game_t);

#endif