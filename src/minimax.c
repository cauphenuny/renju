// author: Cauphenuny
// date: 2024/09/21

#include "board.h"
#include "game.h"
#include "util.h"
#include "zobrist.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int evaluate_pos(board_t board, point_t pos, int sgn)
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

static int evaluate(board_t board, int sgn)
{
    int sum = 0;
    for (int x = 0; x < BOARD_SIZE; x++) {
        for (int y = 0; y < BOARD_SIZE; y++) {
            if (!board[x][y]) continue;
            sum += evaluate_pos(board, (point_t){x, y}, sgn);
        }
    }
    return sum;
}

typedef struct {
    board_t board;
    zobrist_t hash;
    int value;
    int id;
    int result;
    point_t pos;
} mmstate_t;

static mmstate_t mm_put_piece(mmstate_t state, point_t pos, int id)
{
    mmstate_t ret = state;
    int put_id = id == 1 ? 1 : 2;
    ret.hash = zobrist_update(ret.hash, pos, ret.board[pos.x][pos.y], put_id);
    ret.board[pos.x][pos.y] = put_id;
    ret.pos = pos;
    ret.value = evaluate(ret.board, id);
    ret.result = check(ret.board, pos);
    // print(ret.board);
    // getchar();
    ret.id = -id;
    return ret;
}

static mmstate_t mm_remove_piece(mmstate_t state, point_t pos)
{
    mmstate_t ret = state;
    ret.hash = zobrist_update(ret.hash, pos, ret.board[pos.x][pos.y], 0);
    ret.board[pos.x][pos.y] = 0;
    ret.pos = pos;
    ret.value = 0;
    ret.result = check(ret.board, pos);
    return ret;
}

static bool adjacent(board_t board, point_t pos)
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
} result_t;

typedef struct {
    int depth;
    result_t result;
#ifdef TEST
    board_t board;
#endif
} cache_t;

static mmstate_t cur_state;

static int eval_reuse_cnt;

#define EVAL_CACHE_SIZE       10000019
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

static int tim, time_limit;

static unsigned int eval_cache_hash(zobrist_t key)
{
    return key % EVAL_CACHE_SIZE;
}

static eval_cache_t create_eval_cache()
{
    eval_cache_t map;
    size_t table_size = sizeof(eval_cache_entry_t*) * EVAL_CACHE_SIZE;
    map.table = (eval_cache_entry_t**)malloc(table_size);
    memset(map.table, 0, table_size);
    return map;
}

static cache_t* eval_cache_insert(eval_cache_t map, zobrist_t key, cache_t value)
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

static int max_cnt = 0;

static cache_t* eval_cache_search(eval_cache_t map, zobrist_t key)
{
    unsigned int index = eval_cache_hash(key);
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

static void free_eval_cache(eval_cache_t map)
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

static result_t minimax_search(int depth, int alpha, int beta)
{
    if (depth == 0 || cur_state.result || get_time(tim) > time_limit)
        return (result_t){cur_state.value, cur_state.pos};
    // print(abstate.board);
    // log("depth = %d, alpha = %d, beta = %d, is_max = %d", depth, alpha, beta,
    // is_max); prompt_pause(); log("depth = %d", depth);
    cache_t* entry = eval_cache_search(eval_cache, cur_state.hash);
    if (entry != NULL) {
#ifdef TEST
        if (!is_equal(entry->board, cur_state.board)) {
            log_e("zobrist hash conflict!");
            print(entry->board);
            print(cur_state.board);
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
        memcpy(new_cache.board, cur_state.board, sizeof(board_t));
#endif
        entry = eval_cache_insert(eval_cache, cur_state.hash, new_cache);
    }
    int id = cur_state.id;
    result_t ret = {-0x7f7f7f7f * id, {-1, -1}};
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            point_t pos = (point_t){i, j};
            if (adjacent(cur_state.board, pos) &&
                !is_forbidden(cur_state.board, pos, id == 1 ? 1 : 2, false)) {
                cur_state = mm_put_piece(cur_state, pos, id);
                result_t child = minimax_search(depth - 1, alpha, beta);
                cur_state = mm_remove_piece(cur_state, pos);
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

point_t minimax(const game_t game, void* assets)
{
    (void)assets;

    max_cnt = 0;
    time_limit = game.time_limit - 10;
    if (game.count == 0) return (point_t){BOARD_SIZE / 2, BOARD_SIZE / 2};
    tim = record_time();
    eval_cache = create_eval_cache();
    eval_reuse_cnt = 0, eval_cache_size = 0;
    if (eval_cache_buffer == NULL) {
        eval_cache_buffer = (eval_cache_entry_t*)malloc(
            sizeof(eval_cache_entry_t) * EVAL_CACHE_ENTRY_SIZE);
    }
    memcpy(cur_state.board, game.board, sizeof(board_t));
    cur_state.id = game.cur_id == 1 ? 1 : -1;
    cur_state.pos = game.steps[game.count - 1];
    cur_state.result = 0;
    cur_state.value = evaluate(cur_state.board, -cur_state.id);
    point_t pos;
    for (int8_t i = 0; i < BOARD_SIZE; i++) {
        for (int8_t j = 0; j < BOARD_SIZE; j++) {
            point_t p = {i, j};
            if (available(game.board, p) && !is_forbidden(game.board, p, game.cur_id, false)) {
                pos = p;
                break;
            }
        }
    }
    int maxdepth = 0;
    log("searching...");
    for (int i = game.cur_id;; i += 2) {
        result_t ret = minimax_search(i, -0x7f7f7f7f, 0x7f7f7f7f);
        if (get_time(tim) < time_limit - 10 && inboard(ret.pos))
            pos = ret.pos, maxdepth = i;
        else
            break;
    }
    free_eval_cache(eval_cache);
    log("max hash conflict: %d", max_cnt);
    log("maxdepth %d, total %d, reused %d, speed: %.2lf", maxdepth, eval_cache_size, eval_reuse_cnt,
        (double)eval_cache_size / get_time(tim));
    assert(inboard(pos) && available(game.board, pos));
    return pos;
}