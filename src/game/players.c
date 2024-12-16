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
#ifdef NO_FORBID
    .check_depth = 0,  // TODO: change to 1
#else
    .check_depth = 1,
#endif
    .network = NULL,
    .eval_type = NONE, 
};
static mcts_param_t mcts_params_nn, mcts_params_adv;

const bool use_external_eval = 1, use_internal_eval = 0;

player_t preset_players[MAX_PLAYERS] = {
    [MANUAL] = {"human", input_manually, NULL},                 //
    [MCTS] = {"AI (MCTS)", mcts, &mcts_params_default},         //
    [MCTS_ADV] = {"AI (MCTS, advanced)", mcts, &mcts_params_adv},         //
    [MCTS_NN] = {"AI (MCTS, NN)", mcts_nn, &mcts_params_nn},    //
    [MINIMAX] = {"AI (minimax)", minimax, &use_external_eval},  //
    [NEURAL_NETWORK] = {"AI (pure NN)", nn_move, NULL}};

point_t move(game_t game, player_t player) { return player.move(game, player.assets); }

void player_init()
{
    mcts_params_nn = mcts_params_default;
    mcts_params_nn.eval_type = NETWORK;
    mcts_params_adv = mcts_params_default;
    mcts_params_adv.eval_type = TRIVIAL;
}

void bind_network(network_t* network, bool is_train)
{
    preset_players[NEURAL_NETWORK].assets = network;
    mcts_params_nn.network = network;
    mcts_params_nn.is_train = is_train;
}

void bind_output_prob(pfboard_t output_array)
{
    mcts_params_default.output_prob = output_array;
    mcts_params_nn.output_prob = output_array;
}