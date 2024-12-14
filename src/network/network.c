#include "network.h"

#include "board.h"
#include "dataset.h"
#include "game.h"
#include "neuro.h"
#include "util.h"

#include <string.h>

#define N BOARD_SIZE

int predict_sum_time, predict_cnt;

prediction_t predict(const network_t* network,  //
                     const board_t board, point_t last_move, int first_id, int cur_id)
{
    int tim = record_time();
    // log("weight mean: %f", mean(network->shared.conv1.weight, 2 * 32 * 3 * 3));
    // log("bias mean: %f", mean(network->shared.conv1.bias, 32));
    prediction_t prediction = {0};
    sample_input_t input = to_sample_input(board, last_move, first_id, cur_id);
    static float output[2][MAX_CHANNEL * N * N], shared_output[MAX_CHANNEL * N * N];
    memset(output, 0, sizeof(output));
    memset(shared_output, 0, sizeof(shared_output));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            output[0][(i * N + j) + 0 * N * N] = input.p1_pieces[i][j];
            output[0][(i * N + j) + 1 * N * N] = input.p2_pieces[i][j];
            output[0][(i * N + j) + 2 * N * N] = input.current_player;
        }
    }
    output[0][(last_move.x * N + last_move.y) + 3 * N * N] = 1;
    int cur = 0;
    size2d_t size = {N, N};

    conv2d(output[cur], output[1 - cur], size, network->shared.conv1, network_params.shared.conv1,
           relu),
        cur = 1 - cur;
    conv2d(output[cur], output[1 - cur], size, network->shared.conv2, network_params.shared.conv2,
           relu),
        cur = 1 - cur;
    conv2d(output[cur], output[1 - cur], size, network->shared.conv3, network_params.shared.conv3,
           relu),
        cur = 1 - cur;
    memcpy(shared_output, output[cur], sizeof(shared_output));

    conv2d(output[cur], output[1 - cur], size, network->value.conv, network_params.value.conv,
           relu),
        cur = 1 - cur;
    linear(output[cur], output[1 - cur], network->value.linear, network_params.value.linear,
           softmax),
        cur = 1 - cur;
    prediction.eval = output[cur][2] - output[cur][0];

    memcpy(output[cur], shared_output, sizeof(shared_output));
    conv2d(output[cur], output[1 - cur], size, network->policy.conv1, network_params.policy.conv1,
           relu),
        cur = 1 - cur;
    conv2d(output[cur], output[1 - cur], size, network->policy.conv2, network_params.policy.conv2,
           relu),
        cur = 1 - cur;
    static float stored_output[N * N];
    memcpy(stored_output, output[cur], sizeof(stored_output));  // Residual Layer
    linear(output[cur], output[1 - cur], network->policy.linear, network_params.policy.linear,
           NULL),
        cur = 1 - cur;
    for (int i = 0; i < N * N; i++) output[cur][i] += stored_output[i];  // Residual Layer
    softmax(output[cur], N * N);
    memcpy(prediction.prob, output[cur], sizeof(prediction.prob));
    // log("consumption: %dms", get_time(tim));
    predict_sum_time += get_time(tim);
    predict_cnt++;
    // if (predict_cnt % 1000 == 0) {
    //     double avg = (double)predict_sum_time / predict_cnt;
    //     if (avg < 10)
    //         log("average calculate time: %.2lfms", avg);
    //     else
    //         log_w("average calculate time: %.2lfms", avg);
    // }
    return prediction;
}

void print_prediction(const prediction_t prediction)
{
    log("eval: %.3lf, entropy: %.3f, prob:", prediction.eval,
        entropy((float*)prediction.prob, N * N, false));
    board_t board = {0};
    print_prob(board, prediction.prob);
}

int save_network(const network_t* network, const char* file_basename)
{
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
    fwrite(network, sizeof(network_t), 1, file);
    fclose(file);
    log("network saved to %s", file_fullname);
    return 0;
}

int load_network(network_t* network, const char* filename)
{
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
    network_params_t get_params;
    fread(&get_params, sizeof(get_params), 1, file);
    if (memcmp(&get_params, &network_params, sizeof(network_params))) {
        log_e("network params mismatch");
        return 3;
    }
    fread(network, sizeof(network_t), 1, file);
    fclose(file);
    log("network loaded from %s", filename);
    return 0;
}

point_t nn_move(game_t game, const void* assets)
{
    if (!game.count) return (point_t){N / 2, N / 2};
    const network_t* network = assets;
    board_t board;
    memcpy(board, game.board, sizeof(board_t));
    prediction_t prediction =
        predict(network, board, game.steps[game.count - 1], game.first_id, game.cur_id);
    point_t pos = {0, 0};
    bool forbid = game.cur_id == game.first_id;
    for (int8_t i = 0; i < N; i++) {
        for (int8_t j = 0; j < N; j++) {
            point_t cur = (point_t){i, j};
            if (game.board[i][j]) continue;
            if (forbid && is_forbidden(game.board, cur, game.cur_id, false)) continue;
            if (prediction.prob[i][j] > prediction.prob[pos.x][pos.y]) {
                pos = cur;
            }
        }
    }
    board[pos.x][pos.y] = game.cur_id;
    return pos;
}