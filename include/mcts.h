#ifndef MCTS_H
#define MCTS_H

#include "board.h"
#include "game.h"
#include "network.h"

enum {
    NONE,
    ADVANCED,
    NETWORK,
};

typedef struct {
    double C_puct;
    int min_time;
    int min_count;
    int8_t wrap_rad;
    int check_depth;
    network_t* network;
    pfboard_t output_prob;
    bool is_train;
    int eval_type;
} mcts_param_t;  // parameters for mcts

point_t mcts(game_t game, const void* assets);
point_t mcts_nn(game_t game, const void* assets);

#endif