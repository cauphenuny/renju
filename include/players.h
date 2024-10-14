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
    PLAYER_CNT,
};

#define MAX_PLAYERS 20

typedef struct {
    const char* name;
    point_t (*move)(const game_t, void* assets);
    void* assets;
} player_t;

extern player_t preset_players[MAX_PLAYERS];

extern int player_cnt;

point_t move(const game_t, player_t);

#endif