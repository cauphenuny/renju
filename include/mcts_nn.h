#ifndef MCTS_NN_H
#define MCTS_NN_H

#include "game.h"
#include "board.h"
#include "neuro.h"

typedef struct {
    neural_network_t* network;
} mcts_nn_param_t;

point_t mcts_nn(game_t game, void* assets);

#endif