#ifndef MCTS_H
#define MCTS_H

#include "board.h"
#include "game.h"
#include "zobrist.h"

typedef struct {
    double C;
    int MIN_TIME;
    int MAX_TIME;
    int MIN_COUNT;
    int8_t WRAP_RAD;
} mcts_parm_t; // parameters for mcts

point_t mcts(const game_t, mcts_parm_t);

#endif