#ifndef MCTS_H
#define MCTS_H

#include "board.h"
#include "game.h"
#include "zobrist.h"

#if BOARD_SIZE <= 16
typedef uint32_t cpr_t;
#elif BOARD_SIZE <= 32
typedef uint64_t cpr_t;
#else
#    error "board size too large!"
#endif

typedef cpr_t cprboard_t[BOARD_SIZE];

typedef struct {
    cprboard_t board;
    cprboard_t visited;
    int piece_cnt;
    int visited_cnt;
    int result;
    int count;
    int8_t id;
    int8_t score;
    point_t pos;
    point_t danger_pos[2];
    zobrist_t hash;
    point_t begin; // wrap left-top
    point_t end; // wrap right-bottom
    int capacity;
} state_t;

typedef struct {
    double C;
    int MIN_TIME;
    int MAX_TIME;
    int MIN_COUNT;
    int8_t WRAP_RAD;
} mcts_parm_t; // parameters for mcts

typedef struct edge_t {
    struct node_t* to;
    struct edge_t* next;
} edge_t;

typedef struct node_t {
    state_t state;
    int son_cnt, parent_cnt;
    struct edge_t *son_edge, *parent_edge;
} node_t;

typedef struct hash_entry_t {
    zobrist_t key;
    node_t* value;
    struct hash_entry_t* next;
} hash_entry_t;

typedef struct {
    hash_entry_t** table;
} hash_map_t;

typedef struct {
    int tot, edge_tot, hash_tot, reused_tot;
    int first_id;
    mcts_parm_t mcts_parm;
    state_t last_status;
    hash_map_t hash_map;
} mcts_assets_t;

point_t mcts(const game_t, mcts_assets_t*);
void assets_init(mcts_assets_t*);

#endif