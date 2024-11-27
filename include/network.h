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

const static struct {
    conv_params_t conv;
    linear_params_t linear;
    max_pool_params_t max_pool;
} checker_params = {
    .conv = {1, 32, 5, 2},
    .max_pool = {32, 5, 5},
    .linear = {288, 3},
};

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

typedef struct {
    CONV(conv, 1, 32, 5, 2);
    LINEAR(linear, 288, 3);
} checker_network_t;

const static struct {
    struct {
        conv_params_t conv1, conv2, conv3;
    } shared;
    struct {
        conv_params_t conv;
        linear_params_t linear1, linear2;
    } value;
    struct {
        conv_params_t conv1, conv2;
    } policy;
} predictor_params = {
    .shared = {{2, 32, 3, 1},      //
               {32, 64, 3, 1},     //
               {64, 256, 5, 2}},   //
    .value = {{256, 4, 1, 0},      //
              {4 * 15 * 15, 128},  //
              {128, 1}},           //
    .policy = {{256, 16, 5, 2},    //
               {16, 1, 1, 0}},     //
};

typedef struct {
    struct {
        CONV(conv1, 2, 32, 3, 1);
        CONV(conv2, 32, 64, 3, 1);
        CONV(conv3, 64, 256, 5, 2);
    } shared;
    struct {
        CONV(conv, 256, 4, 1, 0);
        LINEAR(linear1, 4 * 15 * 15, 128);
        LINEAR(linear2, 128, 1);
    } value;
    struct {
        CONV(conv1, 256, 16, 5, 2);
        CONV(conv2, 16, 1, 1, 0);
    } policy;
} predictor_network_t;

#undef CONV
#undef LINEAR

checker_network_t checker_load(const char* file_name);
int checker_forward(const checker_network_t* network, const board_t board);
void checker_save(const checker_network_t* network, const char* file_name);

typedef struct {
    double eval;
    fboard_t prob;
} prediction_t;

prediction_t predict(const predictor_network_t* predictor, const board_t board, int first_id,
                     int cur_id);
void print_prediction(const prediction_t prediction);
int predictor_load(predictor_network_t* network, const char* file_name);
int predictor_save(const predictor_network_t* network, const char* file_name);

point_t move_nn(game_t game, const void* assets);

#endif