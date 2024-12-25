#ifndef NETWORK_H
#define NETWORK_H

#include "board.h"
#include "game.h"
#include "layer.h"

#define NETWORK_VERSION 7
#define MAX_CHANNEL     256

#include "layer.h"

/*

               [input, 4, 15, 15]
                     |
               [res1, 4:32]
                     |
               [res2, 32:256]
                     |
               [res3, 256:256]
                     |
               [res4, 256:256]
                     |
     +---------------+----------------+
     |                                |
[res, 256:16]                    [conv, 256:16]
     |                                |
[linear, softmax, 16*225:225]    [linear1, relu, 16*225:256]
     |                                |
(policy)                         [linear2, tanh, 256:3]
                                      |
                                   (value)

*/

typedef struct {
    struct {
        residual_block_param_t res1, res2, res3, res4;
    } shared;
    struct {
        residual_block_param_t res;
        linear_params_t linear;
    } policy;
    struct {
        conv_params_t conv;
        linear_params_t linear1, linear2;
    } value;
} network_params_t;

const static network_params_t network_params = {
    .shared =
        {
            .res1 = {.input_channel = 4, .output_channel = 32},
            .res2 = {.input_channel = 32, .output_channel = 256},
            .res3 = {.input_channel = 256, .output_channel = 256},
            .res4 = {.input_channel = 256, .output_channel = 256},
        },
    .policy =
        {
            .res = {.input_channel = 256, .output_channel = 16},
            .linear = {.input_size = 16 * 15 * 15, .output_size = 225, .activate = ACT_SOFTMAX},
        },
    .value =
        {
            .conv = {.input_channel = 256, .output_channel = 16, .kernel_size = 3, .padding = 1},
            .linear1 = {.input_size = 16 * 15 * 15, .output_size = 256, .activate = ACT_RELU},
            .linear2 = {.input_size = 256, .output_size = 3, .activate = ACT_SOFTMAX},
        },
};

typedef struct {
    struct {
        residual_block_t res1, res2, res3, res4;
    } shared;
    struct {
        residual_block_t res;
        linear_layer_t linear;
    } policy;
    struct {
        conv_layer_t conv;
        linear_layer_t linear1, linear2;
    } value;
} network_t;

typedef struct {
    double eval;
    fboard_t prob;
} prediction_t;

void network_init(network_t* network);
void network_free(network_t* network);
int network_load(network_t* network, const char* file_basename);
int network_save(const network_t* network, const char* file_basename);
void forward(const network_t* network, tensor_t* input, tensor_t* policy_output,
             tensor_t* value_output);

prediction_t predict(const network_t* network,  //
                     const board_t board, point_t last_move, int cur_id);
void print_prediction(const prediction_t prediction);

point_t nn_move(game_t game, const void* assets);

#endif