#include "game.h"
#include "board.h"
#include "neuro.h"

typedef struct {
    neural_network_t* network;
} mcts_nn_parm_t;

point_t mcts_nn(game_t game, mcts_nn_parm_t parm);
