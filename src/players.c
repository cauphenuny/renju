// author: Cauphenuny
// date: 2024/07/26

#include "players.h"

#include "board.h"
#include "game.h"
#include "manual.h"
#include "mcts.h"
#include "minimax.h"

#include <string.h>

mcts_param_t mcts_params_default = {
    .C = 1.414,
    .start_c = 1.414,
    .end_c = 1.414,
    .min_time = 200,
    .min_count = 3000,
    .wrap_rad = 2,
    .check_forbid = true,
    .dynamic_area = false,
    .simulate_on_good_pos = false,
    .network = NULL,
};
static mcts_param_t mcts_params_botzone, mcts_params_nn, mcts_params_test;

player_t preset_players[MAX_PLAYERS] = {
    [MANUAL] = {"human", manual, NULL},                                        //
    [MCTS] = {"AI (MCTS)", mcts, &mcts_params_default},                        //
    [MCTS_BZ] = {"AI (MCTS, without checking)", mcts, &mcts_params_botzone},   //
    [MCTS_NN] = {"AI (MCTS, with neural network)", mcts_nn, &mcts_params_nn},  //
    [MCTS_TS] = {"AI (MCTS, test)", mcts, &mcts_params_test},                  //
    [MINIMAX] = {"AI (minimax)", minimax, NULL},                               //
};

point_t move(game_t game, player_t player) { return player.move(game, player.assets); }

void player_init()
{
    mcts_params_botzone = mcts_params_default;
    mcts_params_botzone.check_forbid = false;

    mcts_params_nn = mcts_params_default;

    mcts_params_test = mcts_params_default;
    mcts_params_test.start_c = 3;
    mcts_params_test.end_c = 0.5;
}