#ifndef EVAL_H
#define EVAL_H
#include "board.h"
#include "pattern.h"

typedef struct {
    point_t pos;
    point_t dir;
    pattern_t pattern;
    int id;
} threat_t;

typedef vector_t* threat_storage_t[PAT_TYPE_SIZE];  // vector<threat_t>

#define EVAL_INF 1000000

int eval_pos(board_t board, point_t pos);
int eval_empty_pos(board_t board, point_t pos, int id);
int eval(board_t board);

/// @param storage array of vector<threat_t>*, bind pattern type to storage
void scan_threats(board_t board, int id, int filter_id, threat_storage_t storage);
/// @param return vector<threat_t>
vector_t scan_five_threats(board_t board, int id);
/// @param return vector<threat_t>
vector_t scan_four_threats(board_t board, int id);
/// @param return vector<threat_t>
vector_t find_threats_by_threshold(board_t board, int id, pattern_t threshold);

int add_with_eval(board_t board, int current_eval, point_t pos, int id);

#endif