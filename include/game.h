#ifndef GAME_H
#define GAME_H

#include "board.h"

#define GAMECTRL_REGRET -2
#define GAMECTRL_EXPORT -3

#ifndef GAME_TIME_LIMIT
#    define GAME_TIME_LIMIT 15000
#endif

typedef struct {
    board_t board;
    point_t steps[BOARD_SIZE * BOARD_SIZE];
    int count;
    int cur_player;
    int first_player;
    int winner;
} game_t;

game_t new_game(int first_player);

void game_add_step(game_t* game, point_t pos);

game_t game_backward(game_t game, int after_step);

void game_print(game_t);

void game_export(game_t, const char* name);

#endif