#include "network.h"

#include "board.h"
#include "dataset.h"
#include "game.h"
#include "neuro.h"
#include "util.h"

#include <string.h>

#define N BOARD_SIZE

int checker_forward(const checker_network_t* checker, const board_t board)
{
    float input[N * N] = {0};
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            switch (board[i][j]) {
                case 1: input[i * N + j] = 1; break;
                case 2: input[i * N + j] = -1; break;
                case 0: input[i * N + j] = 0; break;
                default: log_e("invalid board value: %d at (%d, %d)", board[i][j], i, j); return -1;
            }
        }
    }
    float output[2][64 * N * N] = {0};
    memcpy(output[0], input, sizeof(input));
    int cur = 0;
    size2d_t size = {N, N};
    conv2d(output[cur], output[1 - cur], size, checker->conv, checker_params.conv, relu),
        cur = 1 - cur;
    max_pool(output[cur], output[1 - cur], size, checker_params.max_pool), cur = 1 - cur;
    linear(output[cur], output[1 - cur], checker->linear, checker_params.linear, NULL),
        cur = 1 - cur;
#if DEBUG_LEVEL > 0
    fprintf(stderr, "result: %.4lf, %.4lf, %.4lf\n", output[cur][0], output[cur][1],
            output[cur][2]);
#endif
    int index = 0;
    for (int i = 1; i < 3; i++) {
        if (output[cur][i] > output[cur][index]) index = i;
    }
    return index;
}

void checker_save(const checker_network_t* network, const char* file_name)
{
    FILE* file = fopen(file_name, "wb");
    if (!file) {
        log_e("file open failed: %s", file_name);
        return;
    }
    fwrite(network, sizeof(checker_network_t), 1, file);
    fclose(file);
    log("network saved to %s", file_name);
}

checker_network_t checker_load(const char* file_name)
{
    checker_network_t network;
    FILE* file = fopen(file_name, "rb");
    if (!file) {
        log_e("no such file: %s", file_name);
        return (checker_network_t){0};
    }
    fread(&network, sizeof(checker_network_t), 1, file);
    fclose(file);
    log("network loaded from %s", file_name);
    return network;
}

static int sum_time, cnt;

prediction_t predict(const predictor_network_t* predictor, const board_t board, int first_id,
                     int cur_id)
{
    // log("weight mean: %f", mean(predictor->shared.conv1.weight, 2 * 32 * 3 * 3));
    // log("bias mean: %f", mean(predictor->shared.conv1.bias, 32));
    prediction_t prediction = {0};
    sample_input_t input = to_sample_input(board, first_id, cur_id);
    float output[2][256 * N * N] = {0}, shared_output[256 * N * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            output[0][i * N + j] = input.board[i][j];
            output[0][i * N + j + N * N] = input.cur_id[i][j];
        }
    }
    int cur = 0;
    size2d_t size = {N, N};

    int tim = record_time();
    conv2d(output[cur], output[1 - cur], size, predictor->shared.conv1,
           predictor_params.shared.conv1, relu),
        cur = 1 - cur;
    conv2d(output[cur], output[1 - cur], size, predictor->shared.conv2,
           predictor_params.shared.conv2, relu),
        cur = 1 - cur;
    conv2d(output[cur], output[1 - cur], size, predictor->shared.conv3,
           predictor_params.shared.conv3, relu),
        cur = 1 - cur;
    memcpy(shared_output, output[cur], sizeof(shared_output));

    conv2d(output[cur], output[1 - cur], size, predictor->value.conv, predictor_params.value.conv,
           relu),
        cur = 1 - cur;
    linear(output[cur], output[1 - cur], predictor->value.linear1, predictor_params.value.linear1,
           relu),
        cur = 1 - cur;
    linear(output[cur], output[1 - cur], predictor->value.linear2, predictor_params.value.linear2,
           tanh_),
        cur = 1 - cur;
    prediction.eval = output[cur][0];

    memcpy(output[cur], shared_output, sizeof(shared_output));
    conv2d(output[cur], output[1 - cur], size, predictor->policy.conv1,
           predictor_params.policy.conv1, relu),
        cur = 1 - cur;
    conv2d(output[cur], output[1 - cur], size, predictor->policy.conv2,
           predictor_params.policy.conv2, softmax),
        cur = 1 - cur;
    memcpy(prediction.prob, output[cur], sizeof(prediction.prob));
    // log("consumption: %dms", get_time(tim));
    sum_time += get_time(tim);
    cnt++;
    if (cnt % 1000 == 0) {
        double avg = (double)sum_time / cnt;
        if (avg < 10)
            log("average calculate time: %.2lfms", avg);
        else
            log_w("average calculate time: %.2lfms", avg);
    }
    return prediction;
}

void print_prediction(const prediction_t prediction)
{
    log("eval: %.3lf\n", prediction.eval);
    board_t board = {0};
    probability_print(board, prediction.prob);
}

int predictor_save(const predictor_network_t* network, const char* file_basename)
{
    char file_fullname[1024];
    snprintf(file_fullname, 1024, "%s.v%d.%dch.mod", file_basename, NETWORK_VERSION, MAX_CHANNEL);
    FILE* file = fopen(file_fullname, "wb");
    if (!file) {
        log_e("file open failed: %s", file_fullname);
        return 1;
    }
    int version = NETWORK_VERSION;
    fwrite(&version, sizeof(version), 1, file);
    fwrite(&predictor_params, sizeof(predictor_params), 1, file);
    fwrite(network, sizeof(predictor_network_t), 1, file);
    fclose(file);
    log("network saved to %s", file_fullname);
    return 0;
}

int predictor_load(predictor_network_t* network, const char* file_basename)
{
    char file_fullname[1024];
    snprintf(file_fullname, 1024, "%s.v%d.%dch.mod", file_basename, NETWORK_VERSION, MAX_CHANNEL);
    FILE* file = fopen(file_fullname, "rb");
    if (!file) {
        log_e("no such file: %s", file_fullname);
        return 1;
    }
    int version = 0;
    fread(&version, sizeof(version), 1, file);
    if (version != NETWORK_VERSION) {
        log_e("network version mismatch: %d (expect %d)", version, NETWORK_VERSION);
        return 2;
    }
    struct predictor_params_t get_params;
    fread(&get_params, sizeof(get_params), 1, file);
    if (memcmp(&get_params, &predictor_params, sizeof(predictor_params))) {
        log_e("network params mismatch");
        return 3;
    }
    fread(network, sizeof(predictor_network_t), 1, file);
    fclose(file);
    log("network loaded from %s", file_fullname);
    return 0;
}

point_t move_nn(game_t game, const void* assets)
{
    if (!game.count) return (point_t){N / 2, N / 2};
    const predictor_network_t* predictor = assets;
    board_t board;
    memcpy(board, game.board, sizeof(board_t));
    prediction_t prediction = predict(predictor, board, game.first_id, game.cur_id);
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