#include "assert.h"
#include "layer.h"
#include "network.h"
#include "neuro.h"
#include "util.h"

#include <math.h>
#include <stdlib.h>

void test_linear() {
    linear_params_t param = {.input_size = 3, .output_size = 4};
    linear_layer_t layer;
    linear_layer_init(&layer, param);
    log_l("layer initialized");
    for (int i = 0; i < layer.weight.numel; i++) layer.weight.data[i] = i;
    for (int i = 0; i < layer.bias.numel; i++) layer.bias.data[i] = i;
    print_tensor(&layer.weight);
    print_tensor(&layer.bias);
    tensor_t input = tensor_new(3, -1), output = {0};
    log_l("input initialized");
    for (int i = 0; i < input.numel; i++) input.data[i] = i * 2;
    linear_layer(&layer, &input, &output);
    print_tensor(&output);
    float expected_result[] = {10, 29, 48, 67};
    for (int i = 0; i < output.numel; i++) {
        if (fabsf(output.data[i] - expected_result[i]) > 1e-8) {
            log_l("expected: %.2f, got: %.2f", expected_result[i], output.data[i]);
            exit(EXIT_FAILURE);
        }
    }
    linear_layer_free(&layer);
    tensor_free(&input), tensor_free(&output);
    log_i("linear test passed");
}

void test_conv() {
    conv_params_t param = {.input_channel = 3, .output_channel = 4, .kernel_size = 3, .padding = 1};
    conv_layer_t conv;
    conv2d_layer_init(&conv, param);
    log_l("conv layer initialized");
    for (int i = 0; i < conv.weight.numel; i++) conv.weight.data[i] = i;
    for (int i = 0; i < conv.bias.numel; i++) conv.bias.data[i] = i;
    print_tensor(&conv.weight);
    print_tensor(&conv.bias);
    tensor_t input = tensor_new(3, 4, 4, -1), output = {0};
    log_l("input initialized");
    for (int i = 0; i < input.numel; i++) input.data[i] = i * 2;
    conv2d_layer(&conv, &input, &output, 0);
    print_tensor(&output);
    int expected_shape[] = {4, 4, 4};
    for (int i = 0; i < 3; i++) {
        assert(((int*)output.shape.data)[i] == expected_shape[i]);
    }
    assert(output.numel == 4 * 4 * 4);
    float expected_result[] = {
        9042.,  13506., 14028.,  9270.,   13716., 20394., 21096., 13878., 15660., 23202.,  23904.,
        15678., 10014., 14766.,  15180.,  9906.,  21031., 31975., 33469., 22555., 33643.,  51013.,
        53173., 35749., 39475.,  59653.,  61813., 41437., 27187., 41011., 42397., 28375.,  33020.,
        50444., 52910., 35840.,  53570.,  81632., 85250., 57620., 63290., 96104., 99722.,  67196.,
        44360., 67256., 69614.,  46844.,  45009., 68913., 72351., 49125., 73497., 112251., 117327.,
        79491., 87105., 132555., 137631., 92955., 61533., 93501., 96831., 65313.};
    for (int i = 0; i < output.numel; i++) {
        if (fabsf(output.data[i] - expected_result[i]) > 1e-8) {
            log_l("expected: %.2f, got: %.2f", expected_result[i], output.data[i]);
            exit(EXIT_FAILURE);
        }
    }
    conv2d_layer_free(&conv);
    tensor_free(&input), tensor_free(&output);
    log_i("conv test passed");
}

void test_residual_block() {
    residual_block_param_t param = {.input_channel = 3, .output_channel = 4};
    residual_block_t block;
    residual_block_init(&block, param);
    log_l("residual block initialized");
    for (int i = 0; i < block.conv3x3_1.weight.numel; i++) block.conv3x3_1.weight.data[i] = i;
    for (int i = 0; i < block.conv3x3_1.bias.numel; i++) block.conv3x3_1.bias.data[i] = i;
    for (int i = 0; i < block.conv3x3_2.weight.numel; i++) block.conv3x3_2.weight.data[i] = i;
    for (int i = 0; i < block.conv3x3_2.bias.numel; i++) block.conv3x3_2.bias.data[i] = i;
    for (int i = 0; i < block.conv1x1.weight.numel; i++) block.conv1x1.weight.data[i] = i;
    for (int i = 0; i < block.conv1x1.bias.numel; i++) block.conv1x1.bias.data[i] = i;
    print_tensor(&block.conv3x3_1.weight);
    print_tensor(&block.conv3x3_1.bias);
    print_tensor(&block.conv3x3_2.weight);
    print_tensor(&block.conv3x3_2.bias);
    print_tensor(&block.conv1x1.weight);
    print_tensor(&block.conv1x1.bias);
    tensor_t input = {0}, output = {0};
    tensor_renew(&input, 3, 4, 4, -1);
    log_l("input initialized");
    for (int i = 0; i < input.numel; i++) input.data[i] = i % 2 ? (i + 2) : (-(i * 2));
    residual_block(&block, &input, &output);
    print_tensor(&output);
    int expected_shape[] = {4, 4, 4};
    for (int i = 0; i < 3; i++) {
        assert(((int*)output.shape.data)[i] == expected_shape[i]);
    }
    assert(output.numel == 4 * 4 * 4);
    residual_block_free(&block);
    float expected_result[] = {0., 339071.,  325694.,  312845.,  0., 510833.,  489638.,  469043.,
                               0., 489965.,  469022.,  448751.,  0., 287087.,  273890.,  261437.,
                               0., 811420.,  797548.,  785212.,  0., 1263370., 1241572., 1221598.,
                               0., 1232170., 1210516., 1190974., 0., 749176.,  735160.,  723544.,
                               0., 1283769., 1269402., 1257579., 0., 2015907., 1993506., 1974153.,
                               0., 1974375., 1952010., 1933197., 0., 1211265., 1196430., 1185651.,
                               0., 1756118., 1741256., 1729946., 0., 2768444., 2745440., 2726708.,
                               0., 2716580., 2693504., 2675420., 0., 1673354., 1657700., 1647758.};
    for (int i = 0; i < output.numel; i++) {
        if (fabsf(output.data[i] - expected_result[i]) > 1e-8) {
            log_l("expected: %.2f, got: %.2f", expected_result[i], output.data[i]);
            exit(EXIT_FAILURE);
        }
    }
    tensor_free(&input), tensor_free(&output);
    log_i("residual test passed");
}

void test_network() {
    network_t network;
    network_init(&network);
    board_t board = {0};
    double tim = record_time();
    prediction_t prediction = predict(&network, board, (point_t){7, 7}, 1);
    log_i("tim: %.2lfms", get_time(tim));
    print_prediction(prediction);
    network_free(&network);
}

void test_neuro() {
    test_linear();
    test_conv();
    test_residual_block();
    test_network();
}