#ifndef NETWORK_H
#define NETWORK_H

#include "board.h"
#include "game.h"

typedef struct {
    int x, y;
} size2d_t;

typedef struct {
    int input_channel, output_channel;
    int kernel_size, padding;
} conv_params_t;

typedef struct {
    int input_size, output_size;
} linear_params_t;

typedef struct {
    int channel;
    int kernel_size, stride;
} max_pool_params_t;

#define NETWORK_VERSION 5

#define MAX_CHANNEL 128

typedef struct {
    struct {
        conv_params_t conv1, conv2, conv3;
    } shared;
    struct {
        conv_params_t conv;
        linear_params_t linear;
    } value;
    struct {
        conv_params_t conv1, conv2;
        linear_params_t linear;
    } policy;
} network_params_t;

#define CONV(name, ich, och, ksize, pd)          \
    struct {                                     \
        float weight[ich * och * ksize * ksize]; \
        float bias[och];                         \
    } name
#define LINEAR(name, isize, osize)   \
    struct {                         \
        float weight[isize * osize]; \
        float bias[osize];           \
    } name

static const network_params_t network_params = {
    .shared =
        {
            .conv1 = {4, 32, 5, 2},
            .conv2 = {32, 64, 5, 2},
            .conv3 = {64, MAX_CHANNEL, 3, 1},
        },
    .value =
        {
            .conv = {MAX_CHANNEL, 4, 3, 1},
            .linear = {4 * 15 * 15, 3},
        },
    .policy =
        {
            .conv1 = {MAX_CHANNEL, 32, 3, 1},
            .conv2 = {32, 1, 3, 1},
            .linear = {1 * 15 * 15, 225},
        },
};

typedef struct {
    struct {
        CONV(conv1, 4, 32, 5, 2);
        CONV(conv2, 32, 64, 5, 2);
        CONV(conv3, 64, MAX_CHANNEL, 3, 1);
    } shared;
    struct {
        CONV(conv, MAX_CHANNEL, 4, 3, 1);
        LINEAR(linear, 4 * 15 * 15, 3);
    } value;
    struct {
        CONV(conv1, MAX_CHANNEL, 32, 3, 1);
        CONV(conv2, 32, 1, 3, 1);
        LINEAR(linear, 1 * 15 * 15, 225);
    } policy;
} network_t;

#undef CONV
#undef LINEAR

typedef struct {
    double eval;
    fboard_t prob;
} prediction_t;

prediction_t predict(const network_t* network,  //
                     const board_t board, point_t last_move, int cur_id);
void print_prediction(const prediction_t prediction);
int load_network(network_t* network, const char* file_basename);
int save_network(const network_t* network, const char* file_basename);

point_t nn_move(game_t game, const void* assets);

#endif