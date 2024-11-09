#ifndef PLAYERS_H
#define PLAYERS_H

#include "board.h"
#include "game.h"
#include "mcts.h"

enum {
    MANUAL,
    MCTS,     // default MCTS
    MCTS_BZ,  // for botzone
    MCTS_NN,  // MCTS with neural network
    MCTS_TS, // MCTS, test
    MINIMAX,
    MINIMAX_TS,
    PLAYER_CNT,
};

#define MAX_PLAYERS 20

typedef struct {
    const char* name;
    point_t (*move)(game_t, const void* assets);
    const void* assets;
} player_t;

extern player_t preset_players[MAX_PLAYERS];

extern mcts_param_t mcts_params_default;

point_t move(const game_t, player_t);

void player_init(void);

#endif