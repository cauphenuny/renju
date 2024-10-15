// author: Cauphenuny
// date: 2024/07/26
#include "players.h"

#include "board.h"
#include "game.h"
#include "manual.h"
#include "mcts.h"
#include "mcts_nn.h"
#include "minimax.h"

#include <string.h>

static mcts_param_t mcts_preset = {
    .C = 1.414,
    .start_c = 3,
    .end_c = 0.5,
    .min_time = 200,
    .min_count = 300,
    .wrap_rad = 3,
    .check_ban = true,
};
static mcts_param_t mcts2_preset = {
    .C = 1.414,
    .start_c = 3,
    .end_c = 0.5,
    .min_time = 200,
    .min_count = 300,
    .wrap_rad = 2,
    .check_ban = true,
};

player_t preset_players[MAX_PLAYERS] = {
    [MANUAL] = {"human", manual, NULL},      //
    [MCTS]  = {"AI (MCTS)", mcts, &mcts_preset},  //
    [MCTS2] = {"AI (MCTS, radius 2)", mcts, &mcts2_preset}, //
    [MCTS_NN] = {"AI (MCTS, with NN)", mcts_nn, NULL},    //
    [MINIMAX] = {"AI (minimax)", minimax, NULL},    //
};

point_t move(game_t game, player_t player)
{
    return player.move(game, player.assets);
}