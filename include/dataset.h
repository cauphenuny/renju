#ifndef DATASET_H
#define DATASET_H

#include "board.h"
#include "game.h"

typedef struct {
    cboard_t p1_pieces;     // p1: first player
    cboard_t p2_pieces;     // p2: second player
    int8_t current_player;  // 1 for 1st player, -1 for 2nd player, 0 for terminated
    point_t last_move;      // last move
} sample_input_t;

typedef struct {
    int result;     // 0 / 1 / -1: game result for 1st player
    fboard_t prob;  // probability of each position
} sample_output_t;

typedef struct {
    sample_input_t input;
    sample_output_t output;
} sample_t;

sample_t to_sample(const board_t board, point_t last_move, int first_player, int cur_player,
                   const fboard_t prob, int result);
sample_input_t to_sample_input(const board_t board, point_t last_move, int first_player,
                               int cur_player);
void print_sample(sample_t sample);

typedef struct {
    int size;           // number of samples
    int next_pos;       // next position to add a sample
    int capacity;       // capacity of the dataset
    int sizeof_sample;  // size of a sample
    sample_t* samples;  // array of samples
} dataset_t;

dataset_t new_dataset(int capacity);
void free_dataset(dataset_t* dataset);
void shuffle_dataset(const dataset_t* dataset);
void add_testgames(dataset_t* dataset, const game_result_t* results, int count);
void add_games(dataset_t* dataset, const game_result_t* results, int count);
int save_dataset(const dataset_t* dataset, const char* filename);
int load_dataset(dataset_t* dataset, const char* filename);
sample_t random_sample(const dataset_t* dataset);
sample_t find_sample(const dataset_t* dataset, int index);

#endif