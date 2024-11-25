#ifndef NEURO_H
#define NEURO_H

#include "board.h"
#include "game.h"

const static struct {
    struct {
        int input_channel, output_channel;
        int kernel_size, padding;
    } conv;
    struct {
        int input_size, output_size;
    } linear;
    struct {
        int kernel_size, stride;
    } max_pool;
} checker_params = {
    .conv = {1, 32, 5, 2},
    .max_pool = {5, 5},
    .linear = {288, 3},
};

typedef struct {
    struct {
        float weight[32 * 5 * 5];
        float bias[32];
    } conv;
    struct {
        float weight[288 * 3];
        float bias[3];
    } linear;
} checker_network_t;

typedef struct {
    // TODO:
} predictor_network_t;

typedef struct {
    int8_t board[BOARD_SIZE][BOARD_SIZE];
    int8_t winner;
    int8_t final_winner;
} sample_t;

sample_t to_sample(const board_t board, int winner, int final_winner);
void add_samples(game_t* games, int count);
void export_samples(const char* file_name);
void import_samples(const char* file_name);
int dataset_size();
sample_t random_sample();
sample_t find_sample(int index);

checker_network_t checker_new();
checker_network_t checker_load(const char* file_name);
int checker_forward(checker_network_t* network, const board_t board);
void checker_free(checker_network_t* network);
void checker_save(checker_network_t* network, const char* file_name);

#endif