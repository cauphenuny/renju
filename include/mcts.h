#ifndef MCTS_H
#define MCTS_H

#include "board.h"
#include "game.h"
#include "neuro.h"

typedef struct {
    double C, start_c, end_c;
    int min_time;
    int min_count;
    int8_t wrap_rad;
    bool check_forbid;
    bool dynamic_area;
    bool simulate_on_good_pos;
    neural_network_t* network;
    pfboard_t prob_matrix;
} mcts_param_t;  // parameters for mcts

point_t mcts(const game_t game, void* assets);
point_t mcts_nn(const game_t game, void* assets);

#endif