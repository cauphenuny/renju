#include "neuro.h"

#include "board.h"
#include <string.h>

#define N BOARD_SIZE

input_t to_input(const board_t board, int id)
{
    input_t input = {0};
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (!board[i][j]) continue;
            input.data[i * N + j] = board[i][j] == id ? 1 : -1;
        }
    }
    return input;
}

output_t to_output(const fboard_t prob, int result)
{
    output_t output;
    double sum = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) sum += prob[i][j];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) output.data[i * N + j] = prob[i][j] / sum;
    output.data[N * N] = result;
    return output;
}

output_t forward(neural_network_t* network, input_t input)
{
    (void)network, (void)input;
    //TODO:
    return (output_t){0};
}

prediction_t predict(neural_network_t* network, const board_t board, int id)
{
    prediction_t pred;
    const input_t input = to_input(board, id);
    const output_t output = forward(network, input);
    memcpy(pred.prob, output.data, sizeof(pred.prob));
    pred.eval = output.data[N * N];
    return pred;
}