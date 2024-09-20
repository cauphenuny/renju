#ifndef GAME_H
#define GAME_H

#include "board.h"

#define GAMECTRL_REGRET -2
#define GAMECTRL_EXPORT -3

typedef struct game_t {
    board_t board;
    point_t steps[BOARD_SIZE * BOARD_SIZE];
    int step_cnt;
    int current_id;
    int first_id;
    int winner_id;
    int players[3];
} game_t;

#endif