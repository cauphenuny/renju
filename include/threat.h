#ifndef THREAT_H
#define THREAT_H

#include "board.h"
#include "pattern.h"
#include "vector.h"

typedef struct {
    pattern_t type;
    vector_t consists;
    vector_t defenses;
    point_t action;
    int8_t dx, dy;
} threat_info_t;

void free_threat_info(void* threat_info);

typedef struct threat_tree_node_t {
    board_t board;
    threat_info_t threat;
    struct threat_tree_node_t *parent, *son, *brother;
    bool only_four;
} threat_tree_node_t;

/// @param return vector<threat_info_t>
vector_t scan_threats_info(board_t board, int id, bool only_four);

/// @param return vector<threat_info_t>
vector_t find_threats(board_t board, point_t pos, bool only_four);

/// @param return vector<threat_tree_node_t*>
vector_t get_threat_forest(board_t board, int id, bool only_four);

void print_threat_tree(threat_tree_node_t* root);

vector_t vcf(board_t board, int id);

void print_vcf(vector_t point_array);

#endif