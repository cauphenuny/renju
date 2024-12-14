#ifndef NEURO_H
#define NEURO_H
#include <stdbool.h>

void softmax(float x[], int size);
void relu(float x[], int size);
void tanh_(float x[], int size);
double entropy(const float x[], int size, bool normalize);
void conv2d_impl(const float* restrict input, int input_channel, int input_x, int input_y,
                 float* restrict output, int output_channel, int* output_x, int* output_y,
                 const float* restrict kernel, const float* restrict bias, int kernel_size,
                 int padding, void (*activate)(float[], int));
void max_pool_impl(const float* restrict input, int channel, int input_x, int input_y,
                   float* restrict output, int* output_x, int* output_y, int kernel_size,
                   int stride);
void linear_impl(const float* restrict input, int input_size, float* restrict output,
                 int output_size, const float* restrict weight, const float* restrict bias,
                 void (*activate)(float[], int));

#define conv2d(input, output, size, data, param, activate)                                         \
    conv2d_impl(input, param.input_channel, size.x, size.y, output, param.output_channel, &size.x, \
                &size.y, data.weight, data.bias, param.kernel_size, param.padding, activate)

#define max_pool(input, output, size, param)                                      \
    max_pool_impl(input, param.channel, size.x, size.y, output, &size.x, &size.y, \
                  param.kernel_size, param.stride)

#define linear(input, output, data, param, activate)                                        \
    linear_impl(input, param.input_size, output, param.output_size, data.weight, data.bias, \
                activate)

#endif
