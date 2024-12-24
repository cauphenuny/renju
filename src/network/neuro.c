#include "neuro.h"

#include "util.h"
#include "vector.h"

#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void tensor_free(tensor_t* tensor) {
    vector_free(tensor->shape);
    free(tensor->data);
    tensor->data = NULL;
    tensor->capacity = tensor->numel = 0;
}

void tensor_renew(tensor_t* tensor, int first_dim, ...) {
    if (!tensor->shape.data) {
        tensor->shape = vector_new(int, NULL);
    } else {
        tensor->shape.size = 0;
    }
    va_list args;
    va_start(args, first_dim);
    int numel = 1, dim = first_dim;
    while (dim != -1) {
        vector_push_back(tensor->shape, dim);
        numel *= dim;
        dim = va_arg(args, int);
    }
    va_end(args);
    tensor->numel = numel;
    if (tensor->capacity < numel) {
        tensor->data = (float*)realloc(tensor->data, numel * sizeof(float));
        tensor->capacity = numel;
    }
    memset(tensor->data, 0, numel * sizeof(float));
}

tensor_t tensor_new(int first_dim, ...) {
    tensor_t tensor = {0};
    tensor.shape = vector_new(int, NULL);
    va_list args;
    va_start(args, first_dim);
    int numel = 1, dim = first_dim;
    while (dim != -1) {
        vector_push_back(tensor.shape, dim);
        numel *= dim;
        dim = va_arg(args, int);
    }
    va_end(args);
    tensor.numel = numel;
    tensor.data = (float*)malloc(numel * sizeof(float));
    tensor.capacity = numel;
    memset(tensor.data, 0, numel * sizeof(float));
    return tensor;
}

tensor_t tensor1d_new(int size) { return tensor_new(size, -1); }

tensor_t tensor2d_new(int x, int y) { return tensor_new(x, y, -1); }

tensor_t tensor3d_new(int ch, int x, int y) { return tensor_new(ch, x, y, -1); }

tensor_t tensor4d_new(int d1, int d2, int d3, int d4) { return tensor_new(d1, d2, d3, d4, -1); }

void tensor_clear(tensor_t* tensor) { memset(tensor->data, 0, tensor->numel * sizeof(float)); }

tensor_t tensor_clone(const tensor_t* src) {
    tensor_t tensor = {0};
    tensor.capacity = src->capacity;
    tensor.numel = src->numel;
    tensor.data = (float*)malloc(src->capacity * sizeof(float));
    tensor.shape = vector_clone(src->shape);
    memcpy(tensor.data, src->data, src->numel * sizeof(float));
    return tensor;
}

void tensor_add(tensor_t* tensor, const tensor_t* other) {
    assert(tensor->numel == other->numel);
    for (int i = 0; i < tensor->numel; i++) {
        tensor->data[i] += other->data[i];
    }
}

float tensor_get(const tensor_t* tensor, int first_dim, ...) {
    va_list args;
    va_start(args, first_dim);
    vector_t index = vector_new(int, NULL);
    int dim = first_dim;
    while (dim != -1) {
        vector_push_back(index, dim);
        dim = va_arg(args, int);
    }
    int offset = 0, base = 1;
    for (int i = (int)index.size - 1; i >= 0; i--) {
        int idx = vector_get(int, index, i);
        offset += idx * base;
        base *= vector_get(int, tensor->shape, i);
    }
    vector_free(index);
    va_end(args);
    return tensor->data[offset];
}

void tensor_set(tensor_t* tensor, float value, int first_dim, ...) {
    va_list args;
    va_start(args, first_dim);
    vector_t index = vector_new(int, NULL);
    int dim = first_dim;
    while (dim != -1) {
        vector_push_back(index, dim);
        dim = va_arg(args, int);
    }
    int offset = 0, base = 1;
    for (int i = (int)index.size - 1; i >= 0; i--) {
        int idx = vector_get(int, index, i);
        offset += idx * base;
        base *= vector_get(int, tensor->shape, i);
    }
    vector_free(index);
    va_end(args);
    tensor->data[offset] = value;
}

int int_serialize(char* dest, size_t size, const void* ptr) {
    return snprintf(dest, size, "%d", *(int*)ptr);
}

void print_tensor(const tensor_t* tensor) {
    char buffer[65536], shape_buffer[65536];
    vector_serialize(shape_buffer, 65536, ", ", tensor->shape, int_serialize);
    int offset = snprintf(buffer, 65536, "shape: [%s], data: [", shape_buffer);
    for (int i = 0; i < tensor->numel; i++) {
        offset += snprintf(buffer + offset, 65536 - offset, "%.2f%s", tensor->data[i],
                           i == tensor->numel - 1 ? "" : ", ");
    }
    snprintf(buffer + offset, 65536 - offset, "]");
    log_l("%s", buffer);
}

void softmax_array(float* x, int size) {
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

void softmax(tensor_t* tensor) {
    float* x = tensor->data;
    const int size = tensor->numel;
    float max_val = -1e9, sum = 0;
    for (int i = 0; i < size; i++) {
        max_val = fmax(max_val, x[i]);
    }
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] = log(x[i] / sum);
    }
}

void sigmoid(tensor_t* tensor) {
    float* x = tensor->data;
    const int size = tensor->numel;
    for (int i = 0; i < size; i++) {
        x[i] = 1 / (1 + exp(-x[i]));
    }
}

void relu(tensor_t* tensor) {
    float* x = tensor->data;
    const int size = tensor->numel;
    for (int i = 0; i < size; i++) {
        x[i] = x[i] > 0 ? x[i] : 0;
    }
}

void tanh_(tensor_t* tensor) {
    float* x = tensor->data;
    const int size = tensor->numel;
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

#define CONV2D_GEN(name, kernel_size, padding)                                                   \
    void conv2d_##name##_impl(const float* restrict input, int input_channel, int input_x,       \
                              int input_y, float* restrict output, int output_channel,           \
                              int output_x, int output_y, const float* restrict kernel,          \
                              const float* restrict bias) {                                      \
        const int output_size = output_x * output_y;                                             \
        const int input_size = input_x * input_y;                                                \
        const int kernel_sqrsize = kernel_size * kernel_size;                                    \
        const int kernel_all_ch_size = kernel_sqrsize * input_channel;                           \
        /*_Pragma("omp parallel for")*/ for (int och = 0; och < output_channel; och++) {             \
            for (int ich = 0; ich < input_channel; ich++) {                                      \
                for (int i = 0; i < output_x; i++) {                                             \
                    for (int j = 0; j < output_y; j++) {                                         \
                        const int x = i - padding, y = j - padding;                              \
                        float sum = 0;                                                           \
                        for (int offset_x = 0; offset_x < kernel_size; offset_x++) {             \
                            for (int offset_y = 0; offset_y < kernel_size; offset_y++) {         \
                                const int cur_x = x + offset_x, cur_y = y + offset_y;            \
                                if (cur_x >= 0 && cur_x < input_x && cur_y >= 0 &&               \
                                    cur_y < input_y) {                                           \
                                    const float in =                                             \
                                        input[cur_x * input_y + cur_y + ich * input_size];       \
                                    const float ker =                                            \
                                        kernel[offset_x * kernel_size + offset_y +               \
                                               ich * kernel_sqrsize + och * kernel_all_ch_size]; \
                                    sum += in * ker;                                             \
                                }                                                                \
                            }                                                                    \
                        }                                                                        \
                        output[i * output_y + j + och * output_size] += sum;                     \
                    }                                                                            \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
        for (int och = 0; och < output_channel; och++) {                                         \
            for (int i = 0; i < output_x; i++) {                                                 \
                for (int j = 0; j < output_y; j++) {                                             \
                    output[i * output_y + j + och * output_size] += bias[och];                   \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
    }

CONV2D_GEN(3x3p1, 3, 1)

CONV2D_GEN(1x1, 1, 0)

void conv2d_impl(const float* restrict input, int input_channel, int input_x, int input_y,
                 float* restrict output, int output_channel, int output_x, int output_y,
                 const float* restrict kernel, const float* restrict bias, int kernel_size,
                 int padding) {
    const int output_size = output_x * output_y;
    const int input_size = input_x * input_y;
    const int kernel_sqrsize = kernel_size * kernel_size;
    const int kernel_all_ch_size = kernel_sqrsize * input_channel;
// #pragma omp parallel for
    for (int och = 0; och < output_channel; och++) {
        for (int ich = 0; ich < input_channel; ich++) {
            for (int i = 0; i < output_x; i++) {
                for (int j = 0; j < output_y; j++) {
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
                    output[i * output_y + j + och * output_size] += sum;
                }
            }
        }
    }
    for (int och = 0; och < output_channel; och++) {
        for (int i = 0; i < output_x; i++) {
            for (int j = 0; j < output_y; j++) {
                output[i * output_y + j + och * output_size] += bias[och];
            }
        }
    }
}

void linear_impl(const float* restrict input, int input_size, float* restrict output,
                 int output_size, const float* restrict weight, const float* restrict bias) {
    // log("linear: %d => %d", input_size, output_size);
// #pragma omp parallel for
    for (int i = 0; i < output_size; i++) {
        float sum = 0;
        for (int j = 0; j < input_size; j++) {
            sum += input[j] * weight[i * input_size + j];
        }
        output[i] = sum;
        if (bias) output[i] += bias[i];
    }
    // log("mean: %f", mean(output, output_size));
}
