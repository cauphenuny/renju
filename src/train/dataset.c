// author: Cauphenuny
// date: 2024/11/26

#include "dataset.h"

#include "game.h"
#include "util.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N BOARD_SIZE

sample_t to_sample(const board_t board, int perspective, int current, const fboard_t prob,
                   int winner, int result)
{
    assert(current >= -1 && current <= 1);
    assert(winner >= 0 && winner <= 2);
    assert(perspective >= 0 && perspective <= 2);
    sample_t sample = {0};
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            sample.current_id[i][j] = current;
            if (board[i][j]) sample.board[i][j] = board[i][j] == perspective ? 1 : -1;
        }
    }
    memcpy(sample.prob, prob, sizeof(fboard_t));
    sample.winner = winner;
    sample.result = result;
    return sample;
}

void print_sample(sample_t sample)
{
    board_t board = {0};
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            board[i][j] = sample.board[i][j] == 1 ? 1 : (sample.board[i][j] == -1 ? 2 : 0);
        }
    }
    probability_print(board, sample.prob);
    printf("current: %d\n", sample.current_id[0][0]);
    printf("winner: %d, result: %d\n", sample.winner, sample.result);
}

#define DATASET_SIZE 100000

typedef struct {
    int nitems;
    int size;
    sample_t samples[DATASET_SIZE];
} dataset_t;

static dataset_t dataset = {0, sizeof(sample_t), {0}};

static sample_t rotate(sample_t raw)
{
    sample_t ret;
    memcpy(&ret, &raw, sizeof(sample_t));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            ret.board[j][N - 1 - i] = raw.board[i][j];
            ret.prob[j][N - 1 - i] = raw.prob[i][j];
        }
    }
    return ret;
}
static sample_t reflect_x(sample_t raw)
{
    sample_t ret;
    memcpy(&ret, &raw, sizeof(sample_t));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            ret.board[N - 1 - i][j] = raw.board[i][j];
            ret.prob[N - 1 - i][j] = raw.prob[i][j];
        }
    }
    return ret;
}
static sample_t reflect_y(sample_t raw)
{
    sample_t ret = {0};
    memcpy(&ret, &raw, sizeof(sample_t));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            ret.board[i][N - 1 - j] = raw.board[i][j];
            ret.prob[i][N - 1 - j] = raw.prob[i][j];
        }
    }
    return ret;
}

static void add_sample(sample_t sample)
{
    if (dataset.nitems < DATASET_SIZE) {
        dataset.samples[dataset.nitems++] = sample;
    }
}

void add_samples(game_result_t* results, int count, bool transform)
{
    for (int i = 0; i < count; i++) {
        game_t game = results[i].game;
        game_t tmp = game_new(game.first_id, game.time_limit);
        for (int j = 0; j < game.count; j++) {
            point_t pos = game.steps[j];
            game_add_step(&tmp, pos);
            int id = tmp.cur_id, current;
            int winner = (j == (game.count - 1)) ? results[i].winner : 0;
            sample_t raw_sample;
            if (transform) {
                if (winner) {
                    current = 0;
                } else {
                    current = id == game.first_id ? 1 : -1;
                }
                raw_sample =
                    to_sample(tmp.board, game.first_id, current, results[i].prob[j + 1],  //
                              winner, results[i].winner == game.first_id ? 1 : -1);
            } else {
                raw_sample = to_sample(tmp.board, 1, id == 1 ? 1 : -1, results[i].prob[j + 1],
                                       winner, results[i].winner == game.first_id ? 1 : -1);
            }
            dataset.samples[dataset.nitems++] = raw_sample;
            add_sample(raw_sample);
            if (j > 0) {
                add_sample(rotate(raw_sample));
                add_sample(rotate(rotate(raw_sample)));
                add_sample(rotate(rotate(rotate(raw_sample))));
                add_sample(reflect_x(raw_sample));
                add_sample(reflect_y(raw_sample));
            }
            if (dataset.nitems >= DATASET_SIZE) goto ret;
        }
    }
ret:
    log("added %d games, now %d samples", count, dataset.nitems);
}

int export_samples(const char* file_name)
{
    FILE* file = fopen(file_name, "wb");
    if (!file) {
        log_e("file open failed: %s", file_name);
        return 1;
    }
    for (int i = 0; i < dataset.nitems; i++) {
        int a = rand() % dataset.nitems;
        int b = rand() % dataset.nitems;
        sample_t tmp = dataset.samples[a];
        dataset.samples[a] = dataset.samples[b];
        dataset.samples[b] = tmp;
    }
    fwrite(&dataset.nitems, sizeof(dataset.nitems), 1, file);
    fwrite(&dataset.size, sizeof(dataset.size), 1, file);
    fwrite(&dataset.samples, sizeof(sample_t), dataset.nitems, file);
    fclose(file);
    log("sizeof sample_t: %d", sizeof(sample_t));
    log("exported %d samples to %s", dataset.nitems, file_name);
    return 0;
}

int import_samples(const char* file_name)
{
    FILE* file = fopen(file_name, "rb");
    if (!file) {
        log_e("no such file: %s", file_name);
        return 1;
    }
    fread(&dataset.nitems, sizeof(dataset), 1, file);
    fread(&dataset.size, sizeof(dataset.size), 1, file);
    if (dataset.size != sizeof(sample_t)) {
        log_e("size mismatch: read %d, expected %d", dataset.size, sizeof(sample_t));
        fclose(file);
        return 2;
    }
    fread(&dataset.samples, sizeof(sample_t), dataset.nitems, file);
    fclose(file);
    log("imported %d samples from %s", dataset.nitems, file_name);
    return 0;
}

int dataset_size() { return dataset.nitems; }

void dataset_clear() { dataset.nitems = 0; }

sample_t random_sample() { return dataset.samples[rand() % dataset.nitems]; }

sample_t find_sample(int index)
{
    if (index < dataset.nitems) return dataset.samples[index];
    log_e("index out of range: %d", index);
    return dataset.samples[0];
}
