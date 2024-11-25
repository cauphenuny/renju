#include "neuro.h"

#include "board.h"
#include "game.h"
#include "util.h"

#undef log
#include <math.h>
#define log log_l
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N BOARD_SIZE

#define DATASET_SIZE 1048576

sample_t to_sample(const board_t board, int winner, int final_winner)
{
    sample_t sample = {0};
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (!board[i][j]) continue;
            sample.board[i][j] = board[i][j] == 1 ? 1 : -1;
        }
    }
    sample.winner = winner;
    sample.final_winner = final_winner;
    return sample;
}

typedef struct {
    sample_t samples[DATASET_SIZE];
    int size;
} dataset_t;

static dataset_t dataset;

static sample_t rotate(sample_t raw)
{
    sample_t ret = {0};
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            ret.board[j][N - 1 - i] = raw.board[i][j];
        }
    }
    ret.winner = raw.winner;
    ret.final_winner = raw.final_winner;
    return ret;
}
static sample_t reflect_x(sample_t raw)
{
    sample_t ret = {0};
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            ret.board[N - 1 - i][j] = raw.board[i][j];
        }
    }
    ret.winner = raw.winner;
    ret.final_winner = raw.final_winner;
    return ret;
}
static sample_t reflect_y(sample_t raw)
{
    sample_t ret = {0};
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            ret.board[i][N - 1 - j] = raw.board[i][j];
        }
    }
    ret.winner = raw.winner;
    ret.final_winner = raw.final_winner;
    return ret;
}

static void add_sample(sample_t sample)
{
    if (dataset.size < DATASET_SIZE) {
        dataset.samples[dataset.size++] = sample;
    }
}

void add_samples(game_t* games, int count)
{
    for (int i = 0; i < count; i++) {
        game_t tmp = game_new(games[i].first_id, games[i].time_limit);
        for (int j = 0; j < games[i].count; j++) {
            point_t pos = games[i].steps[j];
            game_add_step(&tmp, pos);
            // emphasis_print(tmp.board, pos);
            int id = (j == (games[i].count - 1)) ? games[i].winner : 0;
            sample_t raw_sample = to_sample(tmp.board, id, games[i].winner);
            dataset.samples[dataset.size++] = raw_sample;
            add_sample(raw_sample);
            if (j > 0) {
                add_sample(rotate(raw_sample));
                add_sample(rotate(rotate(raw_sample)));
                add_sample(rotate(rotate(rotate(raw_sample))));
                add_sample(reflect_x(raw_sample));
                add_sample(reflect_y(raw_sample));
            }
            if (dataset.size >= DATASET_SIZE) return;
        }
    }
}

void export_samples(const char* file_name)
{
    FILE* file = fopen(file_name, "wb");
    if (!file) {
        log_e("file open failed: %s", file_name);
        return;
    }
    fwrite(&dataset, sizeof(dataset_t), 1, file);
    fclose(file);
    log("exported %d samples", dataset.size);
    log("sizeof sample_t: %d", sizeof(sample_t));
}

void import_samples(const char* file_name)
{
    FILE* file = fopen(file_name, "rb");
    if (!file) {
        log_e("no such file: %s", file_name);
        return;
    }
    fread(&dataset, sizeof(dataset_t), 1, file);
    fclose(file);
    log("imported %d samples", dataset.size);
}

int dataset_size() { return dataset.size; }

sample_t random_sample() { return dataset.samples[rand() % dataset.size]; }

sample_t find_sample(int index)
{
    if (index < dataset.size) return dataset.samples[index];
    log_e("index out of range: %d", index);
    return dataset.samples[0];
}

static void relu(float x[], int size)
{
    for (int i = 0; i < size; i++) {
        x[i] = x[i] > 0 ? x[i] : 0;
    }
}

static void conv2d(float input[], int input_channel, int input_x, int input_y, float output[],
                   int output_channel, int* output_x, int* output_y, float kernel[], float bias[],
                   int kernel_size, int padding)
{
    const int _output_x = input_x - kernel_size + 2 * padding + 1;
    const int _output_y = input_y - kernel_size + 2 * padding + 1;
    const int output_size = _output_x * _output_y;
    const int input_size = input_x * input_y;
    const int kernel_sqrsize = kernel_size * kernel_size;
    const int kernel_all_ch_size = kernel_sqrsize * input_channel;
    memset(output, 0, output_size * output_channel * sizeof(float));
    // log("conv2d: (%d, %d, %d) => (%d, %d, %d)", input_channel, input_x, input_y, output_channel,
    //     _output_x, _output_y);
    for (int och = 0; och < output_channel; och++) {
        for (int ich = 0; ich < input_channel; ich++) {
            // log("channel %d -> %d:", ich, och);
            for (int i = 0; i < _output_x; i++) {
                for (int j = 0; j < _output_y; j++) {
                    // log("  at pos (%d, %d)", i, j);
                    int x = i - padding, y = j - padding;
                    float sum = 0;
                    for (int offset_x = 0; offset_x < kernel_size; offset_x++) {
                        for (int offset_y = 0; offset_y < kernel_size; offset_y++) {
                            int cur_x = x + offset_x, cur_y = y + offset_y;
                            if (cur_x >= 0 && cur_x < input_x && cur_y >= 0 && cur_y < input_y) {
                                float in = input[cur_x * input_y + cur_y + ich * input_size];
                                float ker = kernel[offset_x * kernel_size + offset_y +
                                                   ich * kernel_sqrsize + och * kernel_all_ch_size];
                                sum += in * ker;
                            }
                        }
                    }
                    output[i * _output_y + j + och * output_size] += sum + bias[och];
                }
            }
        }
    }
    *output_x = _output_x;
    *output_y = _output_y;
}

static void max_pool(float input[], int input_channel, int input_x, int input_y, float output[],
                     int* output_x, int* output_y, int kernel_size, int stride)
{
    const int _output_x = (input_x - kernel_size) / stride + 1;
    const int _output_y = (input_y - kernel_size) / stride + 1;
    const int output_size = _output_x * _output_y;
    const int input_size = input_x * input_y;
    for (int ich = 0; ich < input_channel; ich++) {
        for (int i = 0; i < _output_x; i++) {
            for (int j = 0; j < _output_y; j++) {
                float max_val = -1e9;
                for (int offset_x = 0; offset_x < kernel_size; offset_x++) {
                    for (int offset_y = 0; offset_y < kernel_size; offset_y++) {
                        int cur_x = i * stride + offset_x, cur_y = j * stride + offset_y;
                        if (cur_x >= 0 && cur_x < input_x && cur_y >= 0 && cur_y < input_y) {
                            float val = input[cur_x * input_y + cur_y + ich * input_size];
                            max_val = fmax(max_val, val);
                        }
                    }
                }
                output[i * _output_y + j + ich * output_size] = max_val;
            }
        }
    }
    *output_x = _output_x;
    *output_y = _output_y;
}

static void linear(float input[], int input_size, float output[], int output_size, float weight[],
                   float bias[])
{
    for (int i = 0; i < output_size; i++) {
        float sum = 0;
        for (int j = 0; j < input_size; j++) {
            sum += input[j] * weight[i * input_size + j];
        }
        output[i] = sum + bias[i];
    }
}

int checker_forward(checker_network_t* network, const board_t board)
{
    float input[N * N] = {0};
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            switch (board[i][j]) {
                case 1: input[i * N + j] = 1; break;
                case 2: input[i * N + j] = -1; break;
                case 0: input[i * N + j] = 0; break;
                default: log_e("invalid board value: %d at (%d, %d)", board[i][j], i, j); return -1;
            }
        }
    }
    float output[2][64 * N * N] = {0};
    memcpy(output[0], input, sizeof(input));
    int cur = 0;
    int cur_output_x, cur_output_y;
    conv2d(output[cur], checker_params.conv.input_channel, N, N, output[1 - cur],
           checker_params.conv.output_channel, &cur_output_x, &cur_output_y, network->conv.weight,
           network->conv.bias, checker_params.conv.kernel_size, checker_params.conv.padding),
        cur = 1 - cur;
    // log("conv: ");
    // FILE* file = fopen("conv_get.log", "w");
    // for (int ch = 0; ch < checker_params.conv.output_channel; ch++) {
    //     fprintf(file, "========== ch %d ==========\n", ch);
    //     for (int i = 0; i < cur_output_x; i++) {
    //         for (int j = 0; j < cur_output_y; j++) {
    //             float val = output[cur][i * cur_output_y + j + ch * cur_output_x * cur_output_y];
    //             fprintf(file, "%.3lf ", val);
    //         }
    //         fprintf(file, "\n");
    //     }
    // }
    // fclose(file);
    relu(output[cur], cur_output_x * cur_output_y * checker_params.conv.output_channel);
    // FILE* file = fopen("relu_get.log", "w");
    // fprintf(file, "relu: ");
    // for (int ch = 0; ch < checker_params.conv.output_channel; ch++) {
    //     fprintf(file, "========== ch %d ==========\n", ch);
    //     for (int i = 0; i < cur_output_x; i++) {
    //         for (int j = 0; j < cur_output_y; j++) {
    //             float val = output[cur][i * cur_output_y + j + ch * cur_output_x * cur_output_y];
    //             fprintf(file, "%.3lf ", val);
    //         }
    //         fprintf(file, "\n");
    //     }
    // }
    // fclose(file);
    max_pool(output[cur], checker_params.conv.output_channel, cur_output_x, cur_output_y,
             output[1 - cur], &cur_output_x, &cur_output_y, checker_params.max_pool.kernel_size,
             checker_params.max_pool.stride),
        cur = 1 - cur;
    // FILE* file = fopen("max_get.log", "w");
    // fprintf(file, "max: ");
    // for (int ch = 0; ch < checker_params.conv.output_channel; ch++) {
    //     fprintf(file, "========== ch %d ==========\n", ch);
    //     for (int i = 0; i < cur_output_x; i++) {
    //         for (int j = 0; j < cur_output_y; j++) {
    //             float val = output[cur][i * cur_output_y + j + ch * cur_output_x * cur_output_y];
    //             fprintf(file, "%.3lf ", val);
    //         }
    //         fprintf(file, "\n");
    //     }
    // }
    // fclose(file);
    linear(output[cur], cur_output_x * cur_output_y * checker_params.conv.output_channel,
           output[1 - cur], checker_params.linear.output_size, network->linear.weight,
           network->linear.bias),
        cur = 1 - cur;
    fprintf(stderr, "result: %.3lf, %.3lf, %.3lf\n", output[cur][0], output[cur][1],
            output[cur][2]);
    int index = 0;
    for (int i = 1; i < 2; i++) {
        if (output[cur][i] > output[cur][index]) index = i;
    }
    return index;
}

void checker_save(checker_network_t* network, const char* file_name)
{
    FILE* file = fopen(file_name, "wb");
    if (!file) {
        log_e("file open failed: %s", file_name);
        return;
    }
    fwrite(network, sizeof(checker_network_t), 1, file);
    fclose(file);
    log("network saved to %s", file_name);
}

checker_network_t checker_load(const char* file_name)
{
    checker_network_t network;
    FILE* file = fopen(file_name, "rb");
    if (!file) {
        log_e("no such file: %s", file_name);
        return (checker_network_t){0};
    }
    fread(&network, sizeof(checker_network_t), 1, file);
    fclose(file);
    // for (int i = 0; i < 32; i++) {
    //     for (int j = 0; j < 5; j++) {
    //         for (int k = 0; k < 5; k++) {
    //             printf("%.3lf ", network.conv.weight[i * 5 * 5 + j * 5 + k]);
    //         }
    //     }
    //     printf("\n");
    // }
    log("network loaded from %s", file_name);
    // log("first several params:");
    // for (int i = 0; i < 15; i++) {
    //     printf("%.4lf ", network.conv.weight[i]);
    // }
    // printf("\n");
    // board_t board = {0};
    // for (int i = 0; i < 5; i++) {
    //     board[5][i + 5] = 1;
    // }
    // checker_forward(&network, board);
    return network;
}