#ifndef SERVER_H
#define SERVER_H
#include "game.h"
#include "players.h"

game_result_t start_game(player_t p1, player_t p2, int first_id, int time_limit,
                         game_t* start, network_t* network);
#endif