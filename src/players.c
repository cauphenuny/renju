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
    .start_c = 2,
    .end_c = 1,
    .min_time = 200,
    .min_count = 1000,
    .wrap_rad = 2,
    .check_forbid = true,
    .dynamic_area = false,
};
static mcts_param_t mcts2_preset = {
    .C = 1.414,
    .start_c = 2,
    .end_c = 1,
    .min_time = 200,
    .min_count = 1000,
    .wrap_rad = 2,
    .check_forbid = false,
    .dynamic_area = false,
};

player_t preset_players[MAX_PLAYERS] = {
    [MANUAL] = {"human", manual, NULL},                              //
    [MCTS] = {"AI (MCTS)", mcts, &mcts_preset},                      //
    [MCTS2] = {"AI (MCTS, without checking)", mcts, &mcts2_preset},  //
    [MCTS_NN] = {"AI (MCTS, with neural network)", mcts_nn, NULL},   //
    [MINIMAX] = {"AI (minimax)", minimax, NULL},                     //
};

point_t move(game_t game, player_t player)
{
    return player.move(game, player.assets);
}