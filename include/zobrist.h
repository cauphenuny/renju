#ifndef ZOBRIST_H
#define ZOBRIST_H

#include "board.h"

#include <stdint.h>

typedef uint64_t zobrist_t;

zobrist_t zobrist_create(board_t board);

zobrist_t zobrist_update(zobrist_t prev, point_t pos, int prev_id, int now_id);

void zobrist_init(void);

#endif