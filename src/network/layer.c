#include "layer.h"

#include "neuro.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

void conv2d_layer_init(conv_layer_t* layer, conv_params_t param) {
    *layer = (conv_layer_t){.param = param, .weight = {0}, .bias = {0}};
    layer->weight = tensor_new(layer->param.output_channel, layer->param.input_channel,
                               layer->param.kernel_size, layer->param.kernel_size, -1);
    layer->bias = tensor_new(layer->param.output_channel, -1);
}

void conv2d_layer_free(conv_layer_t* layer) {
    tensor_free(&layer->weight);
    tensor_free(&layer->bias);
}

void conv2d_layer_save(const conv_layer_t* layer, FILE* file) {
    fwrite(&layer->param, sizeof(layer->param), 1, file);
    tensor_save(&layer->weight, file);
    tensor_save(&layer->bias, file);
}

void conv2d_layer_load(conv_layer_t* layer, FILE* file) {
    fread(&layer->param, sizeof(layer->param), 1, file);
    tensor_load(&layer->weight, file);
    tensor_load(&layer->bias, file);
}

void conv2d_layer(const conv_layer_t* layer, const tensor_t* input, tensor_t* output, bool accelerate) {
    const int input_channel = layer->param.input_channel;
    const int input_x = get3d_x(input->shape);
    const int input_y = get3d_y(input->shape);
    const int output_channel = layer->param.output_channel;
    const int output_x = input_x - layer->param.kernel_size + 2 * layer->param.padding + 1;
    const int output_y = input_y - layer->param.kernel_size + 2 * layer->param.padding + 1;
    tensor_renew(output, output_channel, output_x, output_y, -1);
    if (accelerate) {
        conv2d_fast_impl(input->data, input_channel, input_x, input_y,      //
                         output->data, output_channel, output_x, output_y,  //
                         layer->weight.data, layer->bias.data,              //
                         layer->param.kernel_size, layer->param.padding);
    } else {
        conv2d_impl(input->data, input_channel, input_x, input_y,      //
                    output->data, output_channel, output_x, output_y,  //
                    layer->weight.data, layer->bias.data,              //
                    layer->param.kernel_size, layer->param.padding);
    }
    if (layer->param.activate) {
        to_activate_func(layer->param.activate)(output);
    }
}

void linear_layer_init(linear_layer_t* layer, linear_params_t param) {
    *layer = (linear_layer_t){.param = param, .weight = {0}, .bias = {0}};
    layer->weight = tensor_new(layer->param.output_size, layer->param.input_size, -1);
    layer->bias = tensor_new(layer->param.output_size, -1);
}

void linear_layer_free(linear_layer_t* layer) {
    tensor_free(&layer->weight);
    tensor_free(&layer->bias);
}

void linear_layer_save(const linear_layer_t* layer, FILE* file) {
    fwrite(&layer->param, sizeof(layer->param), 1, file);
    tensor_save(&layer->weight, file);
    tensor_save(&layer->bias, file);
}

void linear_layer_load(linear_layer_t* layer, FILE* file) {
    fread(&layer->param, sizeof(layer->param), 1, file);
    tensor_load(&layer->weight, file);
    tensor_load(&layer->bias, file);
}

void linear_layer(const linear_layer_t* layer, const tensor_t* input, tensor_t* output) {
    const int input_size = layer->param.input_size;
    const int output_size = layer->param.output_size;
    tensor_renew(output, output_size, -1);
    linear_impl(input->data, input_size, output->data, output_size,  //
                layer->weight.data, layer->bias.data);
    if (layer->param.activate) {
        to_activate_func(layer->param.activate)(output);
    }
}

void residual_block_init(residual_block_t* block, residual_block_param_t param) {
    block->param = param;
    conv_params_t conv1_param = {
        .input_channel = param.input_channel,
        .output_channel = param.output_channel,
        .kernel_size = 3,
        .padding = 1,
        .activate = ACT_RELU,
    };
    conv_params_t conv2_param = {
        .input_channel = param.output_channel,
        .output_channel = param.output_channel,
        .kernel_size = 3,
        .padding = 1,
        .activate = ACT_NONE,
    };
    conv2d_layer_init(&block->conv3x3_1, conv1_param);
    conv2d_layer_init(&block->conv3x3_2, conv2_param);
    if (param.input_channel != param.output_channel) {
        conv_params_t conv1x1_param = {
            .input_channel = param.input_channel,
            .output_channel = param.output_channel,
            .kernel_size = 1,
            .padding = 0,
            .activate = ACT_NONE,
        };
        conv2d_layer_init(&block->conv1x1, conv1x1_param);
    }
}

void residual_block_free(residual_block_t* block) {
    conv2d_layer_free(&block->conv3x3_1);
    conv2d_layer_free(&block->conv3x3_2);
    if (block->param.input_channel != block->param.output_channel) {
        conv2d_layer_free(&block->conv1x1);
    }
}

void residual_block_save(const residual_block_t* block, FILE* file) {
    fwrite(&block->param, sizeof(block->param), 1, file);
    conv2d_layer_save(&block->conv3x3_1, file);
    conv2d_layer_save(&block->conv3x3_2, file);
    if (block->param.input_channel != block->param.output_channel) {
        conv2d_layer_save(&block->conv1x1, file);
    }
}

void residual_block_load(residual_block_t* block, FILE* file) {
    fread(&block->param, sizeof(block->param), 1, file);
    conv2d_layer_load(&block->conv3x3_1, file);
    conv2d_layer_load(&block->conv3x3_2, file);
    if (block->param.input_channel != block->param.output_channel) {
        conv2d_layer_load(&block->conv1x1, file);
    }
}

void residual_block(const residual_block_t* block, const tensor_t* input, tensor_t* output) {
    if (block->param.input_channel == block->param.output_channel) {
        tensor_renew(output, block->param.output_channel, get3d_xy(input->shape), -1);

        tensor_t tmp = tensor_new(block->param.output_channel, get3d_xy(input->shape), -1);
        conv2d_layer(&block->conv3x3_1, input, &tmp, true);
        conv2d_layer(&block->conv3x3_2, &tmp, output, true);
        tensor_free(&tmp);

        tensor_add(output, input);

        relu(output);

    } else {
        tensor_renew(output, block->param.output_channel, get3d_xy(input->shape), -1);

        tensor_t tmp = tensor_new(block->param.output_channel, get3d_xy(input->shape), -1);
        conv2d_layer(&block->conv3x3_1, input, &tmp, true);
        conv2d_layer(&block->conv3x3_2, &tmp, output, true);

        conv2d_layer(&block->conv1x1, input, &tmp, true);
        tensor_add(output, &tmp);
        tensor_free(&tmp);

        relu(output);
    }
}