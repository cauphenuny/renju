#ifndef THREAT_H
#define THREAT_H

#include "board.h"
#include "pattern.h"
#include "vector.h"

/// @note: need to call free_threat_info
typedef struct {
    pattern_t type;
    vector_t consists;
    vector_t defenses;
    point_t action;
    point_t dir;
    int id;
} threat_info_t;

void free_threat_info(void* threat_info);

/// @return vector<point_t>
vector_t vct(bool only_four, board_t board, int id, double time_ms);

#endif