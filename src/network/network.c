/// @file network.c
/// @brief implementation of neural network

#include "network.h"

#include "board.h"
#include "dataset.h"
#include "game.h"
#include "layer.h"
#include "neuro.h"
#include "trivial.h"
#include "util.h"

#include <string.h>

#define N BOARD_SIZE

void network_init(network_t* network) {
    residual_block_init(&network->shared.res1, network_params.shared.res1);
    residual_block_init(&network->shared.res2, network_params.shared.res2);
    residual_block_init(&network->shared.res3, network_params.shared.res3);
    residual_block_init(&network->shared.res4, network_params.shared.res4);

    residual_block_init(&network->policy.res, network_params.policy.res);
    linear_layer_init(&network->policy.linear, network_params.policy.linear);

    conv2d_layer_init(&network->value.conv, network_params.value.conv);
    linear_layer_init(&network->value.linear1, network_params.value.linear1);
    linear_layer_init(&network->value.linear2, network_params.value.linear2);
}

void network_free(network_t* network) {
    residual_block_free(&network->shared.res1);
    residual_block_free(&network->shared.res2);
    residual_block_free(&network->shared.res3);
    residual_block_free(&network->shared.res4);

    residual_block_free(&network->policy.res);
    linear_layer_free(&network->policy.linear);

    conv2d_layer_free(&network->value.conv);
    linear_layer_free(&network->value.linear1);
    linear_layer_free(&network->value.linear2);
}

void forward(const network_t* network, tensor_t* input, tensor_t* policy_output,
             tensor_t* value_output) {
    tensor_t tmp[2] = {{0}, {0}}, shared_output = {0};
    residual_block(&network->shared.res1, input, &tmp[0]);
    residual_block(&network->shared.res2, &tmp[0], &tmp[1]);
    residual_block(&network->shared.res3, &tmp[1], &tmp[0]);
    residual_block(&network->shared.res4, &tmp[0], &tmp[1]);
    shared_output = tensor_clone(&tmp[1]);

    residual_block(&network->policy.res, &shared_output, &tmp[0]);
    linear_layer(&network->policy.linear, &tmp[0], policy_output);

    conv2d_layer(&network->value.conv, &shared_output, &tmp[0], true);
    linear_layer(&network->value.linear1, &tmp[0], &tmp[1]);
    linear_layer(&network->value.linear2, &tmp[1], value_output);

    tensor_free(&shared_output);
    tensor_free(&tmp[0]);
    tensor_free(&tmp[1]);
}

prediction_t predict(const network_t* network,  //
                     const board_t board, point_t last_move, int cur_id) {
    prediction_t prediction = {0};
    tensor_t input_tensor = {0}, policy_tensor = {0}, value_tensor = {0};
    tensor_renew(&input_tensor, network_params.shared.res1.input_channel, N, N, -1);
    sample_input_t input = to_sample_input(board, last_move, 1, cur_id);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            input_tensor.data[(i * N + j) + 0 * N * N] = input.p1_pieces[i][j];
            input_tensor.data[(i * N + j) + 1 * N * N] = input.p2_pieces[i][j];
            input_tensor.data[(i * N + j) + 2 * N * N] = input.current_player;
        }
    }
    input_tensor.data[(last_move.x * N + last_move.y) + 3 * N * N] = 1;

    forward(network, &input_tensor, &policy_tensor, &value_tensor);

    float *policy = policy_tensor.data, *value = value_tensor.data;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            prediction.prob[i][j] = policy[i * N + j];
        }
    }
    prediction.eval = value[1] - value[2];
    tensor_free(&input_tensor);
    tensor_free(&policy_tensor);
    tensor_free(&value_tensor);
    return prediction;
}

void print_prediction(const prediction_t prediction) {
    log_l("eval: %.3lf, entropy: %.3f, prob:", prediction.eval,
          entropy((float*)prediction.prob, N * N, false));
    board_t board = {0};
    print_prob(board, prediction.prob);
}

int network_save(const network_t* network, const char* file_basename) {
    char file_fullname[256];
    snprintf(file_fullname, 256, "%s.v%d.%dch.mod", file_basename, NETWORK_VERSION, MAX_CHANNEL);
    FILE* file = fopen(file_fullname, "wb");
    if (!file) {
        log_e("file open failed: %s", file_fullname);
        return 1;
    }
    int version = NETWORK_VERSION;
    fwrite(&version, sizeof(version), 1, file);
    fwrite(&network_params, sizeof(network_params), 1, file);
    residual_block_save(&network->shared.res1, file);
    residual_block_save(&network->shared.res2, file);
    residual_block_save(&network->shared.res3, file);
    residual_block_save(&network->shared.res4, file);
    residual_block_save(&network->policy.res, file);
    linear_layer_save(&network->policy.linear, file);
    conv2d_layer_save(&network->value.conv, file);
    linear_layer_save(&network->value.linear1, file);
    linear_layer_save(&network->value.linear2, file);
    fclose(file);
    log_l("network saved to %s", file_fullname);
    return 0;
}

int network_load(network_t* network, const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        log_e("no such file: %s", filename);
        return 1;
    }
    int version = 0;
    fread(&version, sizeof(version), 1, file);
    if (version != NETWORK_VERSION) {
        log_e("network version mismatch: %d (expect %d)", version, NETWORK_VERSION);
        return 2;
    }
    network_params_t got_params;
    fread(&got_params, sizeof(got_params), 1, file);
    if (memcmp(&got_params, &network_params, sizeof(network_params))) {
        log_e("network params mismatch");
        return 3;
    }
    residual_block_load(&network->shared.res1, file);
    residual_block_load(&network->shared.res2, file);
    residual_block_load(&network->shared.res3, file);
    residual_block_load(&network->shared.res4, file);
    residual_block_load(&network->policy.res, file);
    linear_layer_load(&network->policy.linear, file);
    conv2d_layer_load(&network->value.conv, file);
    linear_layer_load(&network->value.linear1, file);
    linear_layer_load(&network->value.linear2, file);
    fclose(file);
    log_l("network loaded from %s", filename);
    return 0;
}

point_t nn_move(game_t game, const void* assets) {
    if (!game.count) return (point_t){(int8_t)N / 2, (int8_t)N / 2};
    const nn_player_param_t param = *(nn_player_param_t*)assets;
    if (!param.network) return (point_t){-1, -1};
    if (param.use_vct) {
        point_t pos = trivial_move(game, (double)game.time_limit / 2, false, true);
        if (in_board(pos)) {
            return pos;
        }
    }
    prediction_t prediction =
        predict(param.network, game.board, game.steps[game.count - 1], game.cur_id);
    point_t pos = {0, 0};
    bool forbid = game.cur_id == 1;
    for (int8_t i = 0; i < N; i++) {
        for (int8_t j = 0; j < N; j++) {
            point_t cur = (point_t){i, j};
            if (game.board[i][j]) continue;
            if (forbid && is_forbidden(game.board, cur, game.cur_id, -1)) continue;
            if (prediction.prob[i][j] > prediction.prob[pos.x][pos.y]) {
                pos = cur;
            }
        }
    }
    return pos;
}