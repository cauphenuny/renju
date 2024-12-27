#ifndef MINIMAX_H
#define MINIMAX_H

#include "board.h"
#include "game.h"

typedef struct {
    bool parallel;
    int max_depth;
    struct {
        bool begin_vct; // use VCT search at the beginning of thinking
        bool look_forward; // look-ahead in searching (e.g. claim - o o o o - is a win)
        bool dynamic_depth; // use dynamic depth search (i.e. search deeper when agent just do a dead-4 defense)
        bool narrow_width; // use narrow width search (i.e. search a smaller set of positions on every layer)
    } optim;
    struct {
        int adjacent; // adjacent range of positions to consider around board center
    } strategy;
} minimax_param_t;

point_t minimax(game_t game, const void* assets);

#endif