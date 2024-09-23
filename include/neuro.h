#ifndef NEURO_H
#define NEURO_H

#include "board.h"

typedef struct {
    int x;
} neural_network_t;

typedef struct {
    double prob[BOARD_SIZE][BOARD_SIZE];
    double value;
} prediction_t;

prediction_t predict(neural_network_t*);

#endif