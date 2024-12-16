#ifndef GAME_H
#define GAME_H

#include "board.h"

#define GAMECTRL_WITHDRAW      0x70
#define GAMECTRL_GIVEUP        0x71
#define GAMECTRL_EXPORT        0x72
#define GAMECTRL_SWITCH_PLAYER 0x73
#define GAMECTRL_EVALUATE      0x74

#ifndef GAME_TIME_LIMIT
#    define GAME_TIME_LIMIT 15000
#endif

typedef struct {
    board_t board;
    point_t steps[BOARD_AREA];
    int time_limit;
    int count;
    int cur_id;
} game_t;

typedef struct {
    game_t game;
    fboard_t prob[BOARD_AREA];
    int winner;
} game_result_t;

game_t new_game(int time_limit);
void add_step(game_t* game, point_t pos);
game_t backward(game_t game, int after_step);
void print_game(game_t game);
void serialize_game(game_t game, const char* file);
game_t restore_game(int time_limit, int count, point_t moves[]);

#endif