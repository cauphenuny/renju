#ifndef NEURO_H
#define NEURO_H

#include "board.h"

typedef struct {
    // TODO:
} neural_network_t;

typedef struct {
    double data[BOARD_SIZE * BOARD_SIZE * 2];
} input_t;

typedef struct {
    double data[BOARD_SIZE * BOARD_SIZE + 1];
} output_t;

typedef struct {
    input_t input;
    output_t output;
} sample_t;

typedef struct {
    double prob[BOARD_SIZE][BOARD_SIZE];
    double eval;
} prediction_t;

output_t forward(neural_network_t* network, input_t input);

sample_t to_sample(const board_t board, int perspective, const fboard_t count, int winner);

prediction_t predict(neural_network_t* network, const board_t board, int id);

#endif