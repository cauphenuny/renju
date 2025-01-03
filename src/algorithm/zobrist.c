/// @file zobrist.c
/// @brief implementation of Zobrist hashing

#include "zobrist.h"

#include "board.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>

zobrist_t zobrist_table[BOARD_SIZE][BOARD_SIZE][2];
bool zobrist_initialized = 0;

void zobrist_init() {
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            for (int k = 0; k < 2; k++) {
                zobrist_table[i][j][k] = ((zobrist_t)rand() << 32) | rand();
            }
        }
    }
    zobrist_initialized = 1;
}

zobrist_t zobrist_create(board_t board) {
    assert(zobrist_initialized);
    zobrist_t hash = 0;
    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            const int piece = board[i][j];
            if (piece != 0) {
                hash ^= zobrist_table[i][j][piece - 1];
            }
        }
    }
    return hash;
}

zobrist_t zobrist_update(zobrist_t hash, point_t pos, int prev_id, int now_id) {
    assert(zobrist_initialized);
    if (prev_id != 0) {
        hash ^= zobrist_table[pos.x][pos.y][prev_id - 1];
    }
    if (now_id != 0) {
        hash ^= zobrist_table[pos.x][pos.y][now_id - 1];
    }
    return hash;
}