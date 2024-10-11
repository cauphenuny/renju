#ifndef NEURO_H
#define NEURO_H

#include "board.h"
#define IN_CHANNELS    3
#define OUT_CHANNELS1  32
#define OUT_CHANNELS2  64
#define KERNEL_SIZE    3
#define PADDING        1
#define FC1_INPUT_DIM  (OUT_CHANNELS2 * 15 * 15)
#define FC1_OUTPUT_DIM 128
#define FC2_OUTPUT_DIM (15 * 15 + 1)

typedef struct {
    float conv1_weight[OUT_CHANNELS1 * IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE];
    float conv1_bias[OUT_CHANNELS1];
    float
        conv2_weight[OUT_CHANNELS2 * OUT_CHANNELS1 * KERNEL_SIZE * KERNEL_SIZE];
    float conv2_bias[OUT_CHANNELS2];
    float fc1_weight[FC1_OUTPUT_DIM * FC1_INPUT_DIM];
    float fc1_bias[FC1_OUTPUT_DIM];
    float fc2_weight[FC2_OUTPUT_DIM * FC1_OUTPUT_DIM];
    float fc2_bias[FC2_OUTPUT_DIM];
} neural_network_t;

typedef struct {
    double input[3][BOARD_SIZE][BOARD_SIZE];
} nn_input_t;

typedef struct {
    double prob[BOARD_SIZE][BOARD_SIZE];
} nn_output_t;

typedef struct {
    double prob[BOARD_SIZE][BOARD_SIZE];
    double value;
} prediction_t;

prediction_t predict(neural_network_t*, board_t, int);

#endif