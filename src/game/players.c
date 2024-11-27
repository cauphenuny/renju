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
    .C_puct = 1.414,
    .min_time = 200,
    .min_count = 3000,
    .wrap_rad = 2,
    .check_depth = 1,
    .dynamic_area = false,
    .simulate_on_good_pos = false,
    .network = NULL,
};
static mcts_param_t mcts_params_botzone, mcts_params_nn;

const bool use_external_eval = 1, use_internal_eval = 0;

player_t preset_players[MAX_PLAYERS] = {
    [MANUAL] = {"human", manual, NULL},                                              //
    [MCTS] = {"AI (MCTS)", mcts, &mcts_params_default},                              //
    [MCTS_BZ] = {"AI (MCTS, no check)", mcts, &mcts_params_botzone},  //
    [MCTS_NN] = {"AI (MCTS, NN)", mcts_nn, &mcts_params_nn},        //
    [MINIMAX] = {"AI (minimax)", minimax, &use_external_eval},                       //
    [MINIMAX_TS] = {"AI (minimax, test)", minimax, &use_external_eval},              //
    [NEURAL_NETWORK] = {"AI (pure NN)", move_nn, NULL}};

point_t move(game_t game, player_t player) { return player.move(game, player.assets); }

void player_init()
{
    mcts_params_botzone = mcts_params_default;
    mcts_params_botzone.check_depth = 0;

    mcts_params_nn = mcts_params_default;
}

void bind_network(predictor_network_t* network)
{
    preset_players[NEURAL_NETWORK].assets = network;
    mcts_params_nn.network = network;
}

void bind_output_prob(pfboard_t output_array)
{
    mcts_params_default.output_prob = output_array;
    mcts_params_botzone.output_prob = output_array;
    mcts_params_nn.output_prob = output_array;
}