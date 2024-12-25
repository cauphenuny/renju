#ifndef NEURO_H
#define NEURO_H
#include "vector.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

typedef struct {
    vector_t shape;  // vector<int>
    int capacity, numel;
    float* data;
} tensor_t;

#define get3d_ch(shape) vector_get(int, shape, 0)
#define get3d_x(shape)  vector_get(int, shape, 1)
#define get3d_y(shape)  vector_get(int, shape, 2)
#define get3d_xy(shape) get3d_x(shape), get3d_y(shape)

#define get2d_x(shape)  vector_get(int, shape, 0)
#define get2d_y(shape)  vector_get(int, shape, 1)
#define get2d_xy(shape) get2d_x(shape), get2d_y(shape)

#define get1d(shape) vector_get(int, shape, 0)

void tensor_add(tensor_t* tensor, const tensor_t* other);
void tensor_free(tensor_t* tensor);
tensor_t tensor_new(int first_dim, ...);
tensor_t tensor1d_new(int size);
tensor_t tensor2d_new(int x, int y);
tensor_t tensor3d_new(int ch, int x, int y);
tensor_t tensor4d_new(int d1, int d2, int d3, int d4);
void tensor_renew(tensor_t* tensor, int first_dim, ...);
void tensor_clear(tensor_t* tensor);
tensor_t tensor_clone(const tensor_t* src);
void tensor_set(tensor_t* tensor, float value, int first_dim, ...);
float tensor_get(const tensor_t* tensor, int first_dim, ...);
void tensor_save(const tensor_t* tensor, FILE* file);
void tensor_load(tensor_t* tensor, FILE* file);

void print_tensor(const tensor_t* tensor);

typedef void (*activate_func_t)(tensor_t*);

typedef enum {
    ACT_NONE,
    ACT_SOFTMAX,
    ACT_RELU,
    ACT_TANH,
} activate_t;

activate_func_t to_activate_func(activate_t activate);
void softmax(tensor_t*);
void softmax_array(float*, int size);
void relu(tensor_t*);
void tanh_(tensor_t*);
float mean(float x[], int size);

double entropy(const float x[], int size, bool normalize);
void conv2d_impl(const float* restrict input, int input_channel, int input_x, int input_y,
                 float* restrict output, int output_channel, int output_x, int output_y,
                 const float* restrict kernel, const float* restrict bias, int kernel_size,
                 int padding);
void conv2d_3x3p1_impl(const float* restrict input, int input_channel, int input_x, int input_y,
                       float* restrict output, int output_channel, int output_x, int output_y,
                       const float* restrict kernel, const float* restrict bias);
void conv2d_1x1_impl(const float* restrict input, int input_channel, int input_x, int input_y,
                     float* restrict output, int output_channel, int output_x, int output_y,
                     const float* restrict kernel, const float* restrict bias);
void linear_impl(const float* restrict input, int input_size, float* restrict output,
                 int output_size, const float* restrict weight, const float* restrict bias);

#define conv2d_wrap(input, output, size, data, param, activate)                                    \
    conv2d_impl(input, param.input_channel, size.x, size.y, output, param.output_channel, &size.x, \
                &size.y, data.weight, data.bias, param.kernel_size, param.padding, activate)

#define linear_wrap(input, output, data, param, activate)                                   \
    linear_impl(input, param.input_size, output, param.output_size, data.weight, data.bias, \
                activate)

#endif