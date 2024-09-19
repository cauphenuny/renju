#include "zobrist.h"
#include "board.h"

#include <stdlib.h>
#include <time.h>

static zobrist_t zobrist_table[BOARD_SIZE][BOARD_SIZE][2];

void zobrist_init()
{
    srand(time(NULL));
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            for (int k = 0; k < 2; k++) {
                zobrist_table[i][j][k] = ((zobrist_t)rand() << 32) | rand();
            }
        }
    }
}

zobrist_t zobrist_create(board_t board)
{
    zobrist_t hash = 0;
    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            int piece = board[i][j];
            if (piece != 0) {
                hash ^= zobrist_table[i][j][piece - 1];
            }
        }
    }
    return hash;
}

zobrist_t zobrist_update(zobrist_t hash, point_t pos, int prev_id, int now_id)
{
    if (prev_id != 0) {
        hash ^= zobrist_table[pos.x][pos.y][prev_id - 1];
    }
    if (now_id != 0) {
        hash ^= zobrist_table[pos.x][pos.y][now_id - 1];
    }
    return hash;
}