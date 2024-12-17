// author: Cauphenuny
// date: 2024/09/21

#include "board.h"
#include "eval.h"
#include "game.h"
#include "pattern.h"
#include "threat.h"
#include "trivial.h"
#include "util.h"
#include "vector.h"
#include "zobrist.h"

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    board_t board;
    cboard_t candidate;
    zobrist_t hash;
    long long value;
    int sgn;
    int result;
    point_t pos;
} state_t;

static state_t put_piece(state_t state, point_t pos, int sgn) {
    const int put_id = sgn == 1 ? 1 : 2;
    state.hash = zobrist_update(state.hash, pos, state.board[pos.x][pos.y], put_id);
    state.pos = pos;
    state.value = add_with_eval(state.board, state.value, pos, put_id);
    // state.board[pos.x][pos.y] = put_id, state.value = eval(state.board, NULL);
    state.result = check(state.board, pos);
    state.sgn = -sgn;
    state.candidate[pos.x][pos.y] = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            point_t np = (point_t){pos.x + i, pos.y + j};
            if (in_board(np) && !state.board[np.x][np.y]) {
                state.candidate[np.x][np.y] = 1;
            }
        }
    }
    vector_t threats = find_threats(state.board, pos, false);
    for_each(threat_t, threats, threat) {
        point_t pos = threat.pos;
        state.candidate[pos.x][pos.y] = 1;
    }
    vector_free(threats);
    return state;
}

typedef struct {
    int value;
    point_t pos;
} result_t;

typedef struct {
    int depth;
    result_t result;
#ifdef TEST
    board_t board;
#endif
} cache_t;

static int eval_reuse_cnt;

#define EVAL_CACHE_SIZE       1000003
#define EVAL_CACHE_ENTRY_SIZE EVAL_CACHE_SIZE * 5

typedef struct eval_cache_entry_t {
    zobrist_t key;
    cache_t value;
    struct eval_cache_entry_t* next;
} eval_cache_entry_t;

// eval_cache_entry_t* eval_cache_buffer;

typedef struct {
    eval_cache_entry_t** table;
} eval_cache_t;

static eval_cache_t eval_cache;

static eval_cache_entry_t* eval_cache_buffer;
static int eval_cache_size;

static double tim, time_limit;

static bool use_vct;

static unsigned int eval_cache_hash(zobrist_t key) { return key % EVAL_CACHE_SIZE; }

static eval_cache_t create_eval_cache() {
    eval_cache_t map;
    const size_t table_size = sizeof(eval_cache_entry_t*) * EVAL_CACHE_SIZE;
    map.table = (eval_cache_entry_t**)malloc(table_size);
    memset(map.table, 0, table_size);
    return map;
}

static void init_adjacent(state_t* state) {
    memset(state->candidate, 0, sizeof(state->candidate));
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (!state->board[i][j]) continue;
            point_t pos = (point_t){i, j};
            for (int x = -1; x <= 1; x++) {
                for (int y = -1; y <= 1; y++) {
                    point_t np = (point_t){i + x, j + y};
                    if (in_board(np) && !state->board[np.x][np.y]) {
                        state->candidate[np.x][np.y] = 1;
                    }
                }
            }
        }
    }
}

static cache_t* eval_cache_insert(eval_cache_t map, zobrist_t key, cache_t value) {
    const unsigned int index = eval_cache_hash(key);
    eval_cache_entry_t* new_entry = eval_cache_buffer + eval_cache_size++;
    if (eval_cache_size > EVAL_CACHE_ENTRY_SIZE) return NULL;
    new_entry->key = key;
    new_entry->value = value;
    new_entry->next = map.table[index];
    map.table[index] = new_entry;
    return &(map.table[index]->value);
}

static int max_cnt = 0;

static cache_t* eval_cache_search(eval_cache_t map, zobrist_t key) {
    const unsigned int index = eval_cache_hash(key);
    eval_cache_entry_t* entry = map.table[index];
    int cnt = 0;
    while (entry != NULL) {
        if (entry->key == key) {
            return &(entry->value);
        }
        entry = entry->next;
        cnt++;
    }
    max_cnt = max(max_cnt, cnt);
    return NULL;
}

static void free_eval_cache(eval_cache_t map) {
    // for (size_t i = 0; i < EVAL_CACHE_SIZE; i++) {
    //     eval_cache_entry_t* entry = map.table[i];
    //     while (entry != NULL) {
    //         eval_cache_entry_t* temp = entry;
    //         entry = entry->next;
    //         free(temp);
    //     }
    // }
    free(map.table);
}

typedef struct {
    pattern_t level;
    point_t pos;
} point_eval_t;

static int eval_cmp(const void* p1, const void* p2) {
    point_eval_t a = *(point_eval_t*)p1, b = *(point_eval_t*)p2;
    if (a.level != b.level) return b.level - a.level;
    return 0;
}

static result_t minimax_search(state_t state, int depth, int alpha, int beta) {
    const int sgn = state.sgn, put_id = sgn == 1 ? 1 : 2;
    if (depth == 0 || state.result || get_time(tim) > time_limit)
        return (result_t){state.value, state.pos};
    cache_t* entry = eval_cache_search(eval_cache, state.hash);
    if (entry != NULL) {
#ifdef TEST
        if (!is_equal(entry->board, state.board)) {
            log_e("zobrist hash conflict!");
            print(entry->board);
            print(state.board);
            prompt_pause();
        }
#endif
        if (entry->depth >= depth) {
            eval_reuse_cnt++;
            return entry->result;
        }
    } else {
        cache_t new_cache = {depth, (result_t){0, (point_t){0, 0}}};
#ifdef TEST
        memcpy(new_cache.board, state.board, sizeof(board_t));
#endif
        entry = eval_cache_insert(eval_cache, state.hash, new_cache);
    }
    result_t ret = {-EVAL_INF * sgn, {-1, -1}};
    vector_t threats = vector_new(threat_t, NULL);
    vector_t available_pos = vector_new(point_eval_t, NULL);
    threat_storage_t storage = {0};
    storage[PAT_WIN] = storage[PAT_A4] = storage[PAT_D4] = storage[PAT_A3] = &threats;
    scan_threats(state.board, put_id, storage);
    scan_threats(state.board, 3 - put_id, storage);
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            point_t pos = (point_t){i, j};
            point_eval_t eval = {
                .level = 0,
                .pos = pos,
            };
            if (state.candidate[pos.x][pos.y] && !is_forbidden(state.board, pos, put_id, 3)) {
                vector_push_back(available_pos, eval);
            }
        }
    }
    const int pat_to_level[PAT_TYPE_SIZE] = {
        [PAT_WIN] = 4, [PAT_A4] = 3, [PAT_D4] = 2, [PAT_A3] = 2, [PAT_D3] = 1,
    };
    for_each(threat_t, threats, threat) {
        for_each_ptr(point_eval_t, available_pos, eval) {
            if (EQUAL_XY(eval->pos, threat.pos)) {
                int level =
                    pat_to_level[threat.id == 3 - put_id ? threat.pattern : threat.pattern - 1];
                eval->level += (1 << level);
                break;
            }
        }
    }
    vector_shuffle(available_pos);
    qsort(available_pos.data, available_pos.size, available_pos.element_size, eval_cmp);
    vector_free(threats);
    for_each(point_eval_t, available_pos, eval) {
        point_t pos = eval.pos;
        result_t child = minimax_search(put_piece(state, pos, sgn), depth - 1, alpha, beta);
        if (sgn == 1) {
            if (child.value > alpha) {
                alpha = child.value;
                ret.pos = pos;
            }
        } else {
            if (child.value < beta) {
                beta = child.value;
                ret.pos = pos;
            }
        }
        if (alpha >= beta) break;
    }
    vector_free(available_pos);
    if (sgn == 1) {
        ret.value = alpha;
    } else {
        ret.value = beta;
    }
    if (entry != NULL) {
        entry->depth = depth;
        entry->result = ret;
    }
    return ret;
}

point_t initial_move(game_t game) {
    point_t pos;
    vector_t better_move = vector_new(threat_t, NULL);
    vector_t normal_move = vector_new(threat_t, NULL);
    vector_t trash = vector_new(threat_t, NULL);
    threat_storage_t storage = {
        [PAT_WIN] = &better_move, [PAT_A4] = &better_move, [PAT_D4] = &better_move,
        [PAT_A3] = &better_move,  [PAT_D3] = &normal_move, [PAT_A2] = &normal_move,
        [PAT_D2] = &trash,
    };
    scan_threats(game.board, game.cur_id, storage);
    scan_threats(game.board, 3 - game.cur_id, storage);
    if (better_move.size) {
        int index = rand() % better_move.size;
        pos = vector_get(threat_t, better_move, index).pos;
    } else if (normal_move.size) {
        int index = rand() % normal_move.size;
        pos = vector_get(threat_t, normal_move, index).pos;
    } else if (trash.size) {
        int index = rand() % trash.size;
        pos = vector_get(threat_t, trash, index).pos;
    } else {
        pos = random_move(game);
    }
    vector_free(better_move), vector_free(normal_move), vector_free(trash);
    return pos;
}

point_t minimax(game_t game, const void* assets) {
    tim = record_time();
    use_vct = *(bool*)assets;
    point_t pos = trivial_move(game, use_vct);
    if (in_board(pos)) {
        return pos;
    } else {
        pos = initial_move(game);
    }

    max_cnt = 0;
    time_limit = game.time_limit * 0.95;
    if (game.count == 0) return (point_t){BOARD_SIZE / 2, BOARD_SIZE / 2};
    eval_cache = create_eval_cache();
    eval_reuse_cnt = 0, eval_cache_size = 0;
    if (eval_cache_buffer == NULL) {
        eval_cache_buffer =
            (eval_cache_entry_t*)malloc(sizeof(eval_cache_entry_t) * EVAL_CACHE_ENTRY_SIZE);
    }
    state_t state = {0};
    memcpy(state.board, game.board, sizeof(board_t));
    init_adjacent(&state);
    state.sgn = game.cur_id == 1 ? 1 : -1;
    state.pos = game.steps[game.count - 1];
    state.result = 0;
    state.value = eval(state.board, NULL);
    int maxdepth = 0;
    result_t best_result = {0};
    log("searching...");
    for (int i = 2; i < 8; i += 2) {
        const result_t ret = minimax_search(state, i, -EVAL_INF, EVAL_INF);
        if (get_time(tim) < time_limit && in_board(ret.pos))
            best_result = ret, maxdepth = i, pos = best_result.pos;
        else
            break;
    }
    free_eval_cache(eval_cache);
    log("max hash conflict: %d", max_cnt);
    log("maxdepth %d, total %d, reused %d, speed: %.2lf", maxdepth, eval_cache_size, eval_reuse_cnt,
        eval_cache_size / get_time(tim));
    assert(in_board(pos) && available(game.board, pos));
    log("evaluate: %d", best_result.value);
    return pos;
}