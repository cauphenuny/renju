#ifndef TRIVIAL_H
#define TRIVIAL_H

#include "game.h"
#include "board.h"

point_t random_move(game_t game);

point_t trivial_move(game_t game, double time_limit, bool use_opening, bool use_vct);

#endif