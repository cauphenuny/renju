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
    .network = NULL,
    .eval_type = NONE,
    .use_vct = true, 
};
static mcts_param_t mcts_params_nn, mcts_params_adv;

const static minimax_param_t minimax_params_easy = {
    .max_depth = 6,
    .use_vct = true,
    .use_parallel = false,
};
const static minimax_param_t minimax_params_normal = {
    .max_depth = 6,
    .use_vct = true,
    .use_parallel = true,
};
const static minimax_param_t minimax_params_hard = {
    .max_depth = 16,
    .use_vct = true,
    .use_parallel = true,
};

player_t preset_players[MAX_PLAYERS] = {
    [MANUAL] = {.name = "human",
                .move = input_manually,
                .assets = NULL,
                .attribute = {.no_time_limit = true}},
    [MCTS] =
        {
            .name = "MCTS",
            .move = mcts,
            .assets = &mcts_params_default,
        },
    [MCTS_ADV] = {.name = "MCTS, VCT",
                  .move = mcts,
                  .assets = &mcts_params_adv,
                  .attribute = {.enable_vct = true}},
    [MCTS_NN] = {.name = "MCTS, NN", .move = mcts_nn, .assets = &mcts_params_nn, .attribute = {}},
    [MINIMAX] =
        {
            .name = "minimax, easy",
            .move = minimax,
            .assets = &minimax_params_easy,
            .attribute = {.enable_vct = minimax_params_easy.use_vct},
        },
    [MINIMAX_VCT] = {.name = "minimax, normal",
                     .move = minimax,
                     .assets = &minimax_params_normal,
                     .attribute = {.enable_vct = minimax_params_normal.use_vct}},
    [MINIMAX_FULL] = {.name = "minimax, hard",
                      .move = minimax,
                      .assets = &minimax_params_hard,
                      .attribute = {.enable_vct = minimax_params_hard.use_vct}},
    [NEURAL_NETWORK] = {.name = "neural network", .move = nn_move, .assets = NULL}};

point_t move(game_t game, player_t player) { return player.move(game, player.assets); }

void player_init() {
    mcts_params_nn = mcts_params_default;
    mcts_params_nn.eval_type = NETWORK;
    mcts_params_adv = mcts_params_default;
    mcts_params_adv.eval_type = ADVANCED;
}

void bind_network(network_t* network, bool is_train) {
    preset_players[NEURAL_NETWORK].assets = network;
    mcts_params_nn.network = network;
    mcts_params_nn.is_train = is_train;
}

void bind_output_prob(pfboard_t output_array) {
    mcts_params_default.output_prob = output_array;
    mcts_params_nn.output_prob = output_array;
}