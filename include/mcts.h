#ifndef MCTS_H
#define MCTS_H

#include "board.h"
#include "game.h"
#include "network.h"

typedef struct {
    double C, start_c, end_c;
    int min_time;
    int min_count;
    int8_t wrap_rad;
    int check_depth;
    bool dynamic_area;
    bool simulate_on_good_pos;
    predictor_network_t* network;
    pfboard_t prob;
} mcts_param_t;  // parameters for mcts

point_t mcts(game_t game, const void* assets);
point_t mcts_nn(game_t game, const void* assets);

#endif