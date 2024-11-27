#ifndef PLAYERS_H
#define PLAYERS_H

#include "board.h"
#include "game.h"
#include "network.h"

enum {
    MANUAL,
    MCTS,     // default MCTS
    MCTS_BZ,  // for botzone
    MCTS_NN,  // MCTS with neural network
    MINIMAX,
    MINIMAX_TS,
    NEURAL_NETWORK,  // pure neural network
    PLAYER_CNT,
};

#define MAX_PLAYERS 20

typedef struct {
    const char* name;
    point_t (*move)(game_t, const void* assets);
    const void* assets;
} player_t;

extern player_t preset_players[MAX_PLAYERS];

point_t move(const game_t, player_t);

void player_init(void);

void bind_network(predictor_network_t*);
void bind_output_prob(pfboard_t output_array);

#endif