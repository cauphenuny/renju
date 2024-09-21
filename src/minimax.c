// author: Cauphenuny <
#include "board.h"
#include "game.h"
#include "util.h"
#include "zobrist.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int ab_evaluate(board_t board, point_t pos, int sgn)
{
    int id = sgn == 1 ? 1 : 2;
    static const int8_t dir[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
    int score = 0;
    for (int k = 0; k < 4; k++) {
        int8_t dx = dir[k][0], dy = dir[k][1];
        for (int8_t offset = -4; offset <= 0; offset++) {
            int cur_score = 0;
            int count = 0;
            int blocked = 0;
            for (int i = 0; i < 5; i++) {
                point_t p = {pos.x + (i + offset) * dx,
                             pos.y + (i + offset) * dy};
                if (inboard(p)) {
                    if (board[p.x][p.y] == id) {
                        count++;
                    } else {
                        if (board[p.x][p.y]) {
                            blocked = 1;
                            break;
                        }
                    }
                }
            }
            if (!blocked) {
                switch (count) {
                case 5: cur_score += 10000; break;
                case 4: cur_score += 1000; break;
                case 3: cur_score += 100; break;
                case 2: cur_score += 10; break;
                case 1: cur_score += 1; break;
                }
            }
            score += cur_score;
        }
    }
    score *= sgn;
    return score;
}

typedef struct {
    board_t board;
    zobrist_t hash;
    int value;
    int id;
    int result;
    point_t pos;
} abstate_t;

abstate_t ab_put_piece(abstate_t state, point_t pos, int id)
{
    abstate_t ret = state;
    int put_id = id == 1 ? 1 : 2;
    ret.hash = zobrist_update(ret.hash, pos, ret.board[pos.x][pos.y], put_id);
    ret.board[pos.x][pos.y] = put_id;
    ret.pos = pos;
    ret.value = ab_evaluate(ret.board, pos, id);
    ret.result = check(ret.board, pos);
    // print(ret.board);
    // getchar();
    ret.id = -id;
    return ret;
}

abstate_t ab_remove_piece(abstate_t state, point_t pos)
{
    abstate_t ret = state;
    ret.hash = zobrist_update(ret.hash, pos, ret.board[pos.x][pos.y], 0);
    ret.board[pos.x][pos.y] = 0;
    ret.pos = pos;
    ret.value = 0;
    ret.result = check(ret.board, pos);
    return ret;
}

bool ab_available(board_t board, point_t pos)
{
    if (board[pos.x][pos.y]) return false;
    point_t np;
    for (np.x = pos.x - 1; np.x <= pos.x + 1; np.x++) {
        for (np.y = pos.y - 1; np.y <= pos.y + 1; np.y++) {
            if (inboard(np) && board[np.x][np.y]) return true;
        }
    }
    return false;
}

typedef struct {
    int value;
    point_t pos;
} abresult_t;

typedef struct {
    int depth;
    abresult_t result;
} abcache_t;

abstate_t abstate;

int eval_reuse_cnt;

#define EVAL_CACHE_SIZE       10000019
#define EVAL_CACHE_ENTRY_SIZE EVAL_CACHE_SIZE * 5

typedef struct eval_cache_entry_t {
    zobrist_t key;
    abcache_t value;
    struct eval_cache_entry_t* next;
} eval_cache_entry_t;

// eval_cache_entry_t* eval_cache_buffer;

typedef struct {
    eval_cache_entry_t** table;
} eval_cache_t;

eval_cache_t eval_cache;

eval_cache_entry_t* eval_cache_buffer;
int eval_cache_size;

int abclock;

unsigned int eval_cache_hash(zobrist_t key)
{
    return key % EVAL_CACHE_SIZE;
}

eval_cache_t create_eval_cache()
{
    eval_cache_t map;
    size_t table_size = sizeof(eval_cache_entry_t*) * EVAL_CACHE_SIZE;
    map.table = (eval_cache_entry_t**)malloc(table_size);
    memset(map.table, 0, table_size);
    return map;
}

abcache_t* eval_cache_insert(eval_cache_t map, zobrist_t key, abcache_t value)
{
    unsigned int index = eval_cache_hash(key);
    eval_cache_entry_t* new_entry = eval_cache_buffer + eval_cache_size++;
    if (eval_cache_size > EVAL_CACHE_ENTRY_SIZE) return NULL;
    new_entry->key = key;
    new_entry->value = value;
    new_entry->next = map.table[index];
    map.table[index] = new_entry;
    return &(map.table[index]->value);
}

abcache_t* eval_cache_search(eval_cache_t map, zobrist_t key)
{
    unsigned int index = eval_cache_hash(key);
    eval_cache_entry_t* entry = map.table[index];
    while (entry != NULL) {
        if (entry->key == key) {
            return &(entry->value);
        }
        entry = entry->next;
    }
    return NULL;
}

void free_eval_cache(eval_cache_t map)
{
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

abresult_t minimax_search(int depth, int alpha, int beta)
{
    if (depth == 0 || abstate.result || get_time(abclock) > GAME_TIME_LIMIT)
        return (abresult_t){abstate.value, abstate.pos};
    // print(abstate.board);
    // log("depth = %d, alpha = %d, beta = %d, is_max = %d", depth, alpha, beta,
    // is_max); prompt_getch(); log("depth = %d", depth);
    abcache_t* entry = eval_cache_search(eval_cache, abstate.hash);
    if (entry != NULL) {
        if (entry->depth >= depth) {
            eval_reuse_cnt++;
            return entry->result;
        }
    } else {
        abcache_t new_cache = {depth, (abresult_t){0, (point_t){0, 0}}};
        entry = eval_cache_insert(eval_cache, abstate.hash, new_cache);
    }
    int id = abstate.id;
    abresult_t ret;
    ret.value = -0x7f7f7f7f * id;
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            point_t pos = (point_t){i, j};
            if (ab_available(abstate.board, pos)) {
                abstate = ab_put_piece(abstate, pos, id);
                abresult_t child = minimax_search(depth - 1, alpha, beta);
                abstate = ab_remove_piece(abstate, pos);
                if (id == 1) {
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
        }
    }
    if (id == 1)
        ret.value = alpha;
    else
        ret.value = beta;
    if (entry != NULL) {
        entry->depth = depth;
        entry->result = ret;
    }
    return ret;
}

point_t minimax(const game_t game)
{
    if (game.step_cnt == 0) return (point_t){BOARD_SIZE / 2, BOARD_SIZE / 2};
    abclock = record_time();
    eval_cache = create_eval_cache();
    eval_reuse_cnt = 0, eval_cache_size = 0;
    if (eval_cache_buffer == NULL) {
        eval_cache_buffer = (eval_cache_entry_t*)malloc(
            sizeof(eval_cache_entry_t) * EVAL_CACHE_ENTRY_SIZE);
    }
    memcpy(abstate.board, game.board, sizeof(board_t));
    abstate.id = game.current_id == 1 ? 1 : -1;
    abstate.pos = game.steps[game.step_cnt - 1];
    abstate.result = 0;
    abstate.value = ab_evaluate(abstate.board, abstate.pos, -abstate.id);
    point_t pos;
    int maxdepth = 0;
    for (int i = 3;; i += 2) {
        abresult_t ret = minimax_search(i, -0x7f7f7f7f, 0x7f7f7f7f);
        if (get_time(abclock) < GAME_TIME_LIMIT)
            pos = ret.pos, maxdepth = i;
        else
            break;
    }
    free_eval_cache(eval_cache);
    log("maxdepth %d, total %d, reused %d", maxdepth, eval_cache_size, eval_reuse_cnt);
    return pos;
}