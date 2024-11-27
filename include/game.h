#ifndef GAME_H
#define GAME_H

#include "board.h"

#define GAMECTRL_WITHDRAW      0x70
#define GAMECTRL_GIVEUP        0x71
#define GAMECTRL_EXPORT        0x72
#define GAMECTRL_CHANGE_PLAYER 0x73

#ifndef GAME_TIME_LIMIT
#    define GAME_TIME_LIMIT 15000
#endif

typedef struct {
    board_t board;
    point_t steps[BOARD_SIZE * BOARD_SIZE];
    int time_limit;
    int count;
    int cur_id;
    int first_id;
} game_t;

typedef struct {
    game_t game;
    fboard_t prob[BOARD_SIZE * BOARD_SIZE];
    int winner;
} game_result_t;

game_t game_new(int first_id, int time_limit);
void game_add_step(game_t* game, point_t pos);
game_t game_backward(game_t game, int after_step);
void game_print(game_t game);
void game_serialize(game_t game, const char* file);
game_t game_import(int time_limit, int first_id, int count, point_t moves[]);

#endif