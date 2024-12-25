#include "neuro.h"

#include "util.h"
#include "vector.h"

#include <assert.h>
#ifdef HAVE_BLAS
#include <cblas.h>
#endif
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

void tensor_save(const tensor_t* tensor, FILE* file) {
    vector_save(&tensor->shape, file);
    fwrite(&tensor->capacity, sizeof(int), 1, file);
    fwrite(&tensor->numel, sizeof(int), 1, file);
    fwrite(tensor->data, sizeof(float), tensor->numel, file);
}

void tensor_load(tensor_t* tensor, FILE* file) {
    vector_load(&tensor->shape, file);
    fread(&tensor->capacity, sizeof(int), 1, file);
    fread(&tensor->numel, sizeof(int), 1, file);
    tensor->data = (float*)malloc(tensor->capacity * sizeof(float));
    fread(tensor->data, sizeof(float), tensor->numel, file);
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

activate_func_t to_activate_func(activate_t activate) {
    switch (activate) {
        case ACT_SOFTMAX: return softmax;
        case ACT_RELU: return relu;
        case ACT_TANH: return tanh_;
        default: return NULL;
    }
}

void softmax(tensor_t* tensor) { softmax_array(tensor->data, tensor->numel); }

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

void conv2d_impl(const float* restrict input, int input_channel, int input_x, int input_y,
                 float* restrict output, int output_channel, int output_x, int output_y,
                 const float* restrict kernel, const float* restrict bias, int kernel_size,
                 int padding) {
    const int output_size = output_x * output_y;
    const int input_size = input_x * input_y;
    const int kernel_sqrsize = kernel_size * kernel_size;
    const int kernel_all_ch_size = kernel_sqrsize * input_channel;
#pragma omp parallel for
    for (int och = 0; och < output_channel; och++) {
        for (int ich = 0; ich < input_channel; ich++) {
            for (int i = 0; i < output_x; i++) {
                for (int j = 0; j < output_y; j++) {
                    const int pos = och * output_size + i * output_y + j;
                    for (int offset_x = 0; offset_x < kernel_size; offset_x++) {
                        for (int offset_y = 0; offset_y < kernel_size; offset_y++) {
                            const int cur_x = i - padding + offset_x,
                                      cur_y = j - padding + offset_y;
                            if (cur_x >= 0 && cur_x < input_x && cur_y >= 0 && cur_y < input_y) {
                                output[pos] +=
                                    input[ich * input_size + cur_x * input_y + cur_y] *
                                    kernel[och * kernel_all_ch_size + ich * kernel_sqrsize +
                                           offset_x * kernel_size + offset_y];
                            }
                        }
                    }
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

// 本函数由 ChatGPT o1 生成
// 函数签名与原先相同
void conv2d_fast_impl(const float* restrict input, int input_channel, int input_x, int input_y,
                      float* restrict output, int output_channel, int output_x, int output_y,
                      const float* restrict kernel, const float* restrict bias, int kernel_size,
                      int padding) {
#ifndef HAVE_BLAS
    return conv2d_impl(input, input_channel, input_x, input_y, output, output_channel, output_x,
                       output_y, kernel, bias, kernel_size, padding);
#else
    // ---------------------------------------------------
    // 1) 预计算一些尺寸
    // ---------------------------------------------------
    const int output_size = output_x * output_y;  // 每个通道上的输出大小
    const int input_size = input_x * input_y;     // 每个通道上的输入大小
    const int kernel_sqrsize = kernel_size * kernel_size;
    // 矩阵乘法需要的形状
    const int M = output_channel;                  // kernel 的行 (OC)
    const int K = input_channel * kernel_sqrsize;  // kernel 的列 (IC*k*k)
    const int N = output_x * output_y;             // im2col 后的列数 (OW*OH)

    // ---------------------------------------------------
    // 2) 准备 im2col 矩阵: 大小 K * N
    //    - K = (input_channel * kernel_size^2)
    //    - N = (output_x * output_y)
    // ---------------------------------------------------
    float* im2col_mat = (float*)calloc(K * N, sizeof(float));
    if (!im2col_mat) {
        fprintf(stderr, "Error: cannot allocate im2col_mat\n");
        return;
    }

    // 生成 im2col_mat
    // im2col_mat 视为 shape=(K, N) 的 row-major 矩阵
    //   row  = [0..K-1], col = [0..N-1]
    //   row = ic * (kernel_size^2) + (offset_x * kernel_size) + offset_y
    //   col = i * output_y + j
    // 其中 (i, j) 遍历输出平面，(offset_x, offset_y) 遍历 kernel 窗口
    // 最后映射到 input[...] 取值

    for (int i = 0; i < output_x; i++) {
        for (int j = 0; j < output_y; j++) {
            const int out_col = i * output_y + j;  // 这一点对应 im2col_mat 的第 out_col 列
            for (int ic = 0; ic < input_channel; ic++) {
                for (int offset_x = 0; offset_x < kernel_size; offset_x++) {
                    for (int offset_y = 0; offset_y < kernel_size; offset_y++) {
                        int cur_x = i - padding + offset_x;
                        int cur_y = j - padding + offset_y;
                        int row_in_k = ic * kernel_sqrsize + offset_x * kernel_size + offset_y;
                        if (cur_x >= 0 && cur_x < input_x && cur_y >= 0 && cur_y < input_y) {
                            im2col_mat[row_in_k * N + out_col] =
                                input[ic * input_size + cur_x * input_y + cur_y];
                        } else {
                            // padding 区域填 0
                            im2col_mat[row_in_k * N + out_col] = 0.0f;
                        }
                    }
                }
            }
        }
    }

    // ---------------------------------------------------
    // 3) 准备 kernel 矩阵 kernel_mat: 大小 M * K
    //    M = output_channel, K = input_channel*kernel_size^2
    // ---------------------------------------------------
    // 如果你的 kernel 本身就是按 (OC, IC, kx, ky) 连续存储的 row-major，
    // 只要保证 "外层OC, 内层IC*k*k" 顺序一致，就可以直接用 kernel 指针当作矩阵了。
    //
    // 不过，有些框架里 kernel 的存储顺序可能与此不一致，需要手动拷贝/重排一下。
    // 这里假设 kernel 已经是 [och, ich, kx, ky] 的 row-major 排布:
    //   kernel[ och * (IC*k*k) + ich * (k*k) + offset_x*k + offset_y ]
    // 刚好对应 (row=och, col=ich*(k*k) + offset_x*k + offset_y)
    // 可以直接把 kernel 当作 M*K 矩阵使用：
    const float* kernel_mat = kernel;
    // 如果你的 kernel 排布并不符合 M*K 的 row-major，就要做一次中间转换。

    // ---------------------------------------------------
    // 4) 调用 BLAS 进行矩阵乘法:  output_mat = kernel_mat * im2col_mat
    //    - kernel_mat (M x K)
    //    - im2col_mat (K x N)
    //    => output_mat (M x N)
    // ---------------------------------------------------
    float* output_mat = (float*)calloc(M * N, sizeof(float));
    if (!output_mat) {
        fprintf(stderr, "Error: cannot allocate output_mat\n");
        free(im2col_mat);
        return;
    }

    // cblas_sgemm 原型 (RowMajor):
    // cblas_sgemm(
    //   CblasRowMajor,
    //   CblasNoTrans, CblasNoTrans,
    //   M,      // 矩阵C的行数
    //   N,      // 矩阵C的列数
    //   K,      // 矩阵A的列数 & 矩阵B的行数
    //   alpha,  // alpha
    //   A, lda, // A是(MxK), lda>=K
    //   B, ldb, // B是(KxN), ldb>=N
    //   beta,   // beta
    //   C, ldc  // C是(MxN), ldc>=N
    // );
    //
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M,           // = output_channel
                N,           // = output_x*output_y
                K,           // = input_channel*(kernel_size^2)
                1.0f,        // alpha
                kernel_mat,  // A
                K,           // lda = K
                im2col_mat,  // B
                N,           // ldb = N
                0.0f,        // beta
                output_mat,  // C
                N            // ldc = N
    );

    // ---------------------------------------------------
    // 5) 给 output_mat (M x N) 加上 bias，再写回到 output
    //    M=output_channel, N=(output_x * output_y)
    // ---------------------------------------------------
    for (int oc = 0; oc < output_channel; oc++) {
        for (int idx = 0; idx < output_size; idx++) {
            output_mat[oc * N + idx] += bias[oc];
        }
    }

    // 此时 output_mat 存储的是按 (oc, i*j) 展开的 2D (M x N)。
    // 只需把 output_mat[..] 搬到 output[..] 即可；
    // 而你的 output 需要的索引是 output[oc * (output_x*output_y) + i*output_y + j].
    // 实际上，这正好是同一个顺序(行 = oc, 列 = i*output_y+j)。可以直接 memcpy。
    memcpy(output, output_mat, sizeof(float) * M * N);

    // ---------------------------------------------------
    // 6) 释放临时内存
    // ---------------------------------------------------
    free(im2col_mat);
    free(output_mat);

#endif
}

void linear_impl(const float* restrict input, int input_size, float* restrict output,
                 int output_size, const float* restrict weight, const float* restrict bias) {
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
}
