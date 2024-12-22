#include "neuro.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void softmax(float x[], int size) {
    // log("softmax: %d", size);
    float max_val = -1e9, sum = 0;
    for (int i = 0; i < size; i++) {
        max_val = fmax(max_val, x[i]);
    }
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void sigmoid(float x[], int size) {
    // log("sigmoid: %d", size);
    for (int i = 0; i < size; i++) {
        x[i] = 1 / (1 + exp(-x[i]));
    }
}

void silu(float x[], int size) {
    // log("silu: %d", size);
    for (int i = 0; i < size; i++) {
        x[i] = x[i] / (1 + exp(-x[i]));
    }
}

void relu(float x[], int size) {
    // log("relu: %d", size);
    for (int i = 0; i < size; i++) {
        x[i] = x[i] > 0 ? x[i] : 0;
    }
}

void tanh_(float x[], int size) {
    // log("tanh: %d", size);
    for (int i = 0; i < size; i++) {
        x[i] = tanh(x[i]);
    }
}

float mean(float x[], int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += x[i];
    }
    return sum / size;
}

double entropy(const float x[], int size, bool normalize) {
    float* arr = (float*)malloc(size * sizeof(float));
    memcpy(arr, x, size * sizeof(float));
    if (normalize) {
        float sum = 0;
        for (int i = 0; i < size; i++) {
            sum += arr[i];
        }
        for (int i = 0; i < size; i++) {
            arr[i] /= sum;
        }
    }
    double ans = 0;
    for (int i = 0; i < size; i++) {
        if (arr[i] > 1e-8) {
            ans -= arr[i] * log(arr[i]);
        }
    }
    free(arr);
    return ans;
}
#define log log_l

void conv2d_impl(const float* restrict input, int input_channel, int input_x, int input_y,
                 float* restrict output, int output_channel, int* output_x, int* output_y,
                 const float* restrict kernel, const float* restrict bias, int kernel_size,
                 int padding, void (*activate)(float[], int)) {
    const int _output_x = input_x - kernel_size + 2 * padding + 1;
    const int _output_y = input_y - kernel_size + 2 * padding + 1;
    const int output_size = _output_x * _output_y;
    const int input_size = input_x * input_y;
    const int kernel_sqrsize = kernel_size * kernel_size;
    const int kernel_all_ch_size = kernel_sqrsize * input_channel;
    memset(output, 0, output_size * output_channel * sizeof(float));
// log("conv2d: (%d, %d, %d) => (%d, %d, %d)", input_channel, input_x, input_y, output_channel,
//     _output_x, _output_y);
#pragma omp parallel for
    for (int och = 0; och < output_channel; och++) {
        for (int ich = 0; ich < input_channel; ich++) {
            for (int i = 0; i < _output_x; i++) {
                for (int j = 0; j < _output_y; j++) {
                    const int x = i - padding, y = j - padding;
                    float sum = 0;
                    for (int offset_x = 0; offset_x < kernel_size; offset_x++) {
                        for (int offset_y = 0; offset_y < kernel_size; offset_y++) {
                            const int cur_x = x + offset_x, cur_y = y + offset_y;
                            if (cur_x >= 0 && cur_x < input_x && cur_y >= 0 && cur_y < input_y) {
                                const float in = input[cur_x * input_y + cur_y + ich * input_size];
                                const float ker =
                                    kernel[offset_x * kernel_size + offset_y +
                                           ich * kernel_sqrsize + och * kernel_all_ch_size];
                                sum += in * ker;
                            }
                        }
                    }
                    output[i * _output_y + j + och * output_size] += sum;
                }
            }
        }
    }
    for (int och = 0; och < output_channel; och++) {
        for (int i = 0; i < _output_x; i++) {
            for (int j = 0; j < _output_y; j++) {
                output[i * _output_y + j + och * output_size] += bias[och];
            }
        }
    }
    if (output_x) *output_x = _output_x;
    if (output_y) *output_y = _output_y;
    if (activate) activate(output, output_channel * output_size);
}

void max_pool_impl(const float* restrict input, int channel, int input_x, int input_y,
                   float* restrict output, int* output_x, int* output_y, int kernel_size,
                   int stride) {
    const int _output_x = (input_x - kernel_size) / stride + 1;
    const int _output_y = (input_y - kernel_size) / stride + 1;
    const int output_size = _output_x * _output_y;
    const int input_size = input_x * input_y;
    // log("max_pool: (%d, %d, %d) => (%d, %d, %d)", channel, input_x, input_y, channel, _output_x,
    //     _output_y);
    for (int ich = 0; ich < channel; ich++) {
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
    if (output_x) *output_x = _output_x;
    if (output_y) *output_y = _output_y;
    // log("mean: %f", mean(output, channel * output_size));
}

void linear_impl(const float* restrict input, int input_size, float* restrict output,
                 int output_size, const float* restrict weight, const float* restrict bias,
                 void (*activate)(float[], int)) {
    // log("linear: %d => %d", input_size, output_size);
#pragma omp parallel for
    for (int i = 0; i < output_size; i++) {
        float sum = 0;
        for (int j = 0; j < input_size; j++) {
            sum += input[j] * weight[i * input_size + j];
        }
        output[i] = sum;
        if (bias) output[i] += bias[i];
    }
    // log("mean: %f", mean(output, output_size));
    if (activate) activate(output, output_size);
    // log("after activation mean: %f", mean(output, output_size));
}

#ifdef TEST
void test_neuro() {
    float kernel[4 * 2 * 3 * 3];
    for (int i = 0; i < 4 * 2 * 3 * 3; i++) kernel[i] = i;
    float input[18];
    for (int i = 0; i < 18; i++) input[i] = i;
    float output[4 * 3 * 3];
    conv2d_impl(input, 2, 3, 3, output, 4, NULL, NULL, kernel, NULL, 3, 1, NULL);
    // for (int i = 0; i < 4 * 3 * 3; i++) {
    //     fprintf(stderr, "%.2f ", output[i]);
    // }
}
#endif