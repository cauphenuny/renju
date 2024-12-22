#ifndef THREAT_H
#define THREAT_H

#include "board.h"
#include "pattern.h"
#include "eval.h"
#include "vector.h"

/// @note: need to call free_threat_info
typedef struct {
    pattern_t type;
    vector_t consists; // vector<point_t>
    vector_t defenses; // vector<point_t>
    point_t action;
    point_t dir;
    int id;
} threat_info_t;

void free_threat_info(void* threat_info);
threat_info_t attach_threat_info(board_t board, threat_t threat);
vector_t find_threats(board_t board, point_t pos, bool only_four);

/// @return vector<point_t>
vector_t vct(bool only_four, board_t board, int id, double time_ms);

#endif