#ifndef TENSOR_H
#define TENSOR_H

#include "neuro.h"

typedef struct {
    int input_channel, output_channel;
    int kernel_size, padding;
    activate_func_t activate_func;
} conv_params_t;

typedef struct {
    conv_params_t param;
    tensor_t weight, bias;
} conv_layer_t;

void conv2d_layer_init(conv_layer_t* layer, conv_params_t param);
void conv2d_layer_free(conv_layer_t* layer);
void conv2d_layer_load(conv_layer_t* layer, FILE* file);
void conv2d_layer_save(const conv_layer_t* layer, FILE* file);
void conv2d_layer(const conv_layer_t* layer, const tensor_t* input, tensor_t* output, int special);

typedef struct {
    int input_size, output_size;
    activate_func_t activate_func;
} linear_params_t;

typedef struct {
    linear_params_t param;
    tensor_t weight, bias;
} linear_layer_t;

void linear_layer_init(linear_layer_t* layer, linear_params_t param);
void linear_layer_free(linear_layer_t* layer);
void linear_layer_load(linear_layer_t* layer, FILE* file);
void linear_layer_save(const linear_layer_t* layer, FILE* file);
void linear_layer(const linear_layer_t* layer, const tensor_t* input, tensor_t* output);

/*
residual block:

            [input, ich]
                |
             (copy)-----------------------------------------------------+
                |                                                       |
[conv3x3_1, ich:och, ksize=3, pd=1]                                     |
                |                                                       |
             (relu)                                [optional, conv1x1, ich:och, ksize=1, pd=0]
                |                                                       |
[conv3x3_2, och:och, ksize=3, pd=1]                                     |
                |                                                       |
              (sum)-----------------------------------------------------+
                |
              (relu)
                |
            [output, och]

*/

typedef struct {
    int input_channel, output_channel;
} residual_block_param_t;

typedef struct {
    residual_block_param_t param;
    conv_layer_t conv3x3_1, conv3x3_2, conv1x1;
} residual_block_t;

void residual_block_init(residual_block_t* block, residual_block_param_t param);
void residual_block_free(residual_block_t* block);
void residual_block_load(residual_block_t* block, FILE* file);
void residual_block_save(const residual_block_t* block, FILE* file);
void residual_block(const residual_block_t* block, const tensor_t* input, tensor_t* output);

#endif