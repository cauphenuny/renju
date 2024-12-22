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
    .eval_type = EVAL_NONE,
    .use_vct = true,
};
static mcts_param_t mcts_params_nn, mcts_params_adv;

const static minimax_param_t minimax_params_normal = {
    .parallel = true,
    .max_depth = 8,
    .strategy = {.adjacent = 1},
    .optim = {.begin_vct = true},
};
const static minimax_param_t minimax_params_advanced = {
    .parallel = true,
    .max_depth = 12,
    .strategy = {.adjacent = 1},
    .optim = {.begin_vct = true, .look_forward = true},
};

const static minimax_param_t minimax_params_ultimate = {
    .parallel = true,
    .max_depth = 12,
    .strategy = {.adjacent = 2},
    .optim = {.begin_vct = true, .look_forward = true},
};

player_t preset_players[MAX_PLAYERS] = {
    [MANUAL] = {.name = "human",
                .move = input_manually,
                .assets = NULL,
                .attribute = {.allow_timeout = true}},
    [MCTS] = {.name = "MCTS", .move = mcts, .assets = &mcts_params_default},
    [MCTS_ADV] = {.name = "MCTS, external VCT",
                  .move = mcts,
                  .assets = &mcts_params_adv,
                  .attribute = {.enable_vct = true}},
    [MCTS_NN] = {.name = "MCTS, NN", .move = mcts_nn, .assets = &mcts_params_nn, .attribute = {}},
    [MINIMAX] = {.name = "minimax, external VCT",
                 .move = minimax,
                 .assets = &minimax_params_normal,
                 .attribute = {.enable_vct = minimax_params_normal.optim.begin_vct}},
    [MINIMAX_ADV] = {.name = "minimax, look forward, external VCT",
                     .move = minimax,
                     .assets = &minimax_params_advanced,
                     .attribute = {.enable_vct = minimax_params_advanced.optim.begin_vct}},
    [MINIMAX_ULT] = {.name = "minimax, look forward, external VCT, larger adjacent",
                     .move = minimax,
                     .assets = &minimax_params_ultimate,
                     .attribute = {.enable_vct = minimax_params_ultimate.optim.begin_vct}},
    [NEURAL_NETWORK] = {.name = "neural network", .move = nn_move, .assets = NULL}};

point_t move(game_t game, player_t player) { return player.move(game, player.assets); }

void player_init() {
    mcts_params_nn = mcts_params_default;
    mcts_params_nn.eval_type = EVAL_NETWORK;
    mcts_params_adv = mcts_params_default;
    mcts_params_adv.eval_type = EVAL_HEURISTIC;
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