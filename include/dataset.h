#ifndef DATASET_H
#define DATASET_H

#include "board.h"
#include "game.h"

typedef struct {
    int8_t board[BOARD_SIZE][BOARD_SIZE];
    int8_t cur_id[BOARD_SIZE][BOARD_SIZE];
} sample_input_t;

typedef struct {
    int8_t board[BOARD_SIZE][BOARD_SIZE];
    int8_t cur_id[BOARD_SIZE][BOARD_SIZE];      // 1 if 1st player, -1 if 2nd player, 0 if
                                                // empty(game terminated)
    int8_t winner;                              // 0 / 1 / 2 : current winner
    int8_t result;                              // 0 / 1 / -1: game reseult for 1st player
    fboard_t prob;                              // probability of each position
} sample_t;

sample_t to_sample(const board_t board, int perspective, int current, const fboard_t prob,
                   int winner, int result);
sample_input_t to_sample_input(const board_t board, int perspective, int current);
void print_sample(sample_t sample);
void add_samples(game_result_t* games, int count, bool transform);
int export_samples(const char* file_name);
int import_samples(const char* file_name);
int dataset_size();
sample_t random_sample();
sample_t find_sample(int index);

#endif