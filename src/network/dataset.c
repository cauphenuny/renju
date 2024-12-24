#include "dataset.h"

#include "board.h"
#include "game.h"
#include "util.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N BOARD_SIZE

sample_input_t to_sample_input(const board_t board, point_t last, int first_player,
                               int cur_player) {
    assert(first_player >= 0 && first_player <= 2);
    assert(cur_player >= 0 && cur_player <= 2);
    sample_input_t input = {0};
    input.last_move = last;
    if (cur_player != 0) {
        cur_player = cur_player == first_player ? 1 : -1;
    }
    input.current_player = cur_player;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (board[i][j]) {
                if (board[i][j] == first_player) {
                    input.p1_pieces[i][j] = 1;
                } else {
                    input.p2_pieces[i][j] = 1;
                }
            }
        }
    }
    return input;
}

sample_t to_sample(const board_t board, point_t last, int first_player, int cur_player,
                   const fboard_t prob, int result) {
    assert(result > 0 && result <= 2);
    sample_t sample = {0};
    sample.input = to_sample_input(board, last, first_player, cur_player);
    memcpy(sample.output.prob, prob, sizeof(fboard_t));
    if (result) {
        sample.output.result = result == first_player ? 1 : -1;
    } else {
        sample.output.result = 0;
    }
    return sample;
}

void print_sample(sample_t sample) {
    board_t board = {0};
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (sample.input.p1_pieces[i][j]) {
                board[i][j] = 1;
            }
            if (sample.input.p2_pieces[i][j]) {
                board[i][j] = 2;
            }
        }
    }
    print_all(board, sample.input.last_move, sample.output.prob);
    printf("current: %d\n", sample.input.current_player);
    printf("result: %d\n", sample.output.result);
}

dataset_t new_dataset(int capacity) {
    dataset_t dataset = {0};
    dataset.capacity = capacity;
    dataset.sizeof_sample = sizeof(sample_t);
    dataset.samples = malloc(sizeof(sample_t) * capacity);
    return dataset;
}

void free_dataset(dataset_t* dataset) {
    if (dataset->samples != NULL) {
        free(dataset->samples);
        dataset->samples = NULL;
    }
    dataset->capacity = dataset->size = dataset->next_pos = 0;
}

void shuffle_dataset(const dataset_t* dataset) {
    const int n = dataset->size;
    for (int i = 0; i < n; i++) {
        int idx = (rand() % (n - i)) + i;
        sample_t tmp = dataset->samples[i];
        dataset->samples[idx] = dataset->samples[i];
        dataset->samples[i] = tmp;
    }
}

static void int_rotate(const cboard_t src, cboard_t dest) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) dest[j][N - 1 - i] = src[i][j];
}

static void int_reflect_x(const cboard_t src, cboard_t dest) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) dest[N - 1 - i][j] = src[i][j];
}

static void int_reflect_y(const cboard_t src, cboard_t dest) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) dest[i][N - 1 - j] = src[i][j];
}

static void float_rotate(const fboard_t src, fboard_t dest) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) dest[j][N - 1 - i] = src[i][j];
}

static void float_reflect_x(const fboard_t src, fboard_t dest) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) dest[N - 1 - i][j] = src[i][j];
}

static void float_reflect_y(const fboard_t src, fboard_t dest) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) dest[i][N - 1 - j] = src[i][j];
}

static sample_t rotate(const sample_t src) {
    sample_t dest;
    dest.input.current_player = src.input.current_player;
    dest.input.last_move = (point_t){src.input.last_move.y, N - 1 - src.input.last_move.x};
    int_rotate(src.input.p1_pieces, dest.input.p1_pieces);
    int_rotate(src.input.p2_pieces, dest.input.p2_pieces);
    float_rotate(src.output.prob, dest.output.prob);
    dest.output.result = src.output.result;
    return dest;
}

static sample_t reflect_x(const sample_t src) {
    sample_t dest;
    dest.input.current_player = src.input.current_player;
    dest.input.last_move = (point_t){N - 1 - src.input.last_move.x, src.input.last_move.y};
    int_reflect_x(src.input.p1_pieces, dest.input.p1_pieces);
    int_reflect_x(src.input.p2_pieces, dest.input.p2_pieces);
    float_reflect_x(src.output.prob, dest.output.prob);
    dest.output.result = src.output.result;
    return dest;
}

static sample_t reflect_y(const sample_t src) {
    sample_t dest;
    dest.input.current_player = src.input.current_player;
    dest.input.last_move = (point_t){src.input.last_move.x, N - 1 - src.input.last_move.y};
    int_reflect_y(src.input.p1_pieces, dest.input.p1_pieces);
    int_reflect_y(src.input.p2_pieces, dest.input.p2_pieces);
    float_reflect_y(src.output.prob, dest.output.prob);
    dest.output.result = src.output.result;
    return dest;
}

static void add_sample(dataset_t* dataset, sample_t sample) {
    dataset->samples[dataset->next_pos++] = sample;
    dataset->size = max(dataset->size, dataset->next_pos);
    if (dataset->next_pos >= dataset->capacity) {
        dataset->next_pos = 0;
    }
}

void add_testgames(dataset_t* dataset, const game_result_t* results, int count) {
    for (int i = 0; i < count; i++) {
        if (!results[i].winner) continue;
        int step_cnt = results[i].game.count;
        game_t tmp = new_game(results[i].game.time_limit);
        for (int j = 0; j < step_cnt; j++) {
            point_t pos = results[i].game.steps[j];
            add_step(&tmp, pos);
            int id = tmp.cur_id;
            int current = (j == step_cnt - 1) ? 0 : id;
            sample_t raw_sample =
                to_sample(tmp.board, pos, 1, current, results[i].prob[j + 1], results[i].winner);
            add_sample(dataset, raw_sample);
        }
    }
    log_l("added %d games, cur pos: %d, size: %d", count, dataset->next_pos, dataset->size);
}

void add_games(dataset_t* dataset, const game_result_t* results, int count) {
    for (int i = 0; i < count; i++) {
        if (!results[i].winner) continue;
        int step_cnt = results[i].game.count;
        game_t tmp = new_game(results[i].game.time_limit);
        for (int j = 0; j < step_cnt; j++) {
            point_t pos = results[i].game.steps[j];
            add_step(&tmp, pos);
            int id = tmp.cur_id;
            int current = (j == step_cnt - 1) ? 0 : id;
            sample_t raw_sample =
                to_sample(tmp.board, pos, 1, current, results[i].prob[j + 1], results[i].winner);
            add_sample(dataset, raw_sample);
            if (j > 2) {
                add_sample(dataset, rotate(raw_sample));
                add_sample(dataset, rotate(rotate(raw_sample)));
                add_sample(dataset, rotate(rotate(rotate(raw_sample))));
                add_sample(dataset, reflect_x(raw_sample));
                add_sample(dataset, reflect_y(raw_sample));
            }
        }
    }
    log_l("added %d games, cur pos: %d, size: %d", count, dataset->next_pos, dataset->size);
}

int save_dataset(const dataset_t* dataset, const char* file_name) {
    FILE* file = fopen(file_name, "wb");
    if (!file) {
        log_e("file open failed: %s", file_name);
        return 1;
    }
    shuffle_dataset(dataset);
    fwrite(&dataset->capacity, sizeof(dataset->capacity), 1, file);
    fwrite(&dataset->size, sizeof(dataset->size), 1, file);
    fwrite(&dataset->next_pos, sizeof(dataset->next_pos), 1, file);
    fwrite(&dataset->sizeof_sample, sizeof(dataset->sizeof_sample), 1, file);
    fwrite(dataset->samples, sizeof(sample_t), dataset->size, file);
    fclose(file);
    log_l("sizeof sample_t: %d", sizeof(sample_t));
    log_l("exported %d samples to %s", dataset->size, file_name);
    return 0;
}

int load_dataset(dataset_t* dataset, const char* file_name) {
    free_dataset(dataset);
    FILE* file = fopen(file_name, "rb");
    if (!file) {
        log_e("no such file: %s", file_name);
        return 1;
    }
    fread(&dataset->capacity, sizeof(dataset->capacity), 1, file);
    fread(&dataset->size, sizeof(dataset->size), 1, file);
    fread(&dataset->next_pos, sizeof(dataset->next_pos), 1, file);
    fread(&dataset->sizeof_sample, sizeof(dataset->sizeof_sample), 1, file);
    dataset->samples = malloc(sizeof(sample_t) * dataset->capacity);
    dataset->size = fread(dataset->samples, sizeof(sample_t), dataset->size, file);
    fclose(file);
    log_l("imported %d samples from %s", dataset->size, file_name);
    // for (int i = 0; i < 10; i++) {
    //     print_sample(random_sample(dataset));
    //     prompt_pause();
    // }
    return 0;
}

sample_t random_sample(const dataset_t* dataset) {
    return dataset->samples[rand() % dataset->size];
}

sample_t find_sample(const dataset_t* dataset, int index) {
    if (index < dataset->size) return dataset->samples[index];
    log_e("index out of range: %d, expected [0, %d)", index, dataset->size);
    return random_sample(dataset);
}
