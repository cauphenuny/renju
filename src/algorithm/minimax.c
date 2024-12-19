#include "minimax.h"

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
    int value;
    int sgn;
    int result;
    point_t pos;
} state_t;

static state_t put_piece(state_t state, point_t pos) {
    const int put_id = state.sgn == 1 ? 1 : 2;
    state.hash = zobrist_update(state.hash, pos, state.board[pos.x][pos.y], put_id);
    state.pos = pos;
    state.value = add_with_eval(state.board, state.value, pos, put_id);
    // state.board[pos.x][pos.y] = put_id, state.value = eval(state.board, NULL);
    state.result = check(state.board, pos);
    if (state.result) state.value = state.result * (EVAL_INF - 1);
    state.sgn = -state.sgn;
    state.candidate[pos.x][pos.y] = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            point_t np = (point_t){pos.x + i, pos.y + j};
            if (!in_board(np) || state.board[np.x][np.y]) continue;
            if (is_forbidden(state.board, np, 3 - put_id, 3)) continue;
            state.candidate[np.x][np.y] = 1;
        }
    }
    // vector_t threats = find_threats(state.board, pos, false);
    // for_each(threat_t, threats, threat) {
    //     point_t p = threat.pos;
    //     state.candidate[p.x][p.y] = 1;
    // }
    // vector_free(threats);
    return state;
}

typedef struct {
    int value;
    point_t pos;
    int tree_size;
} result_t;

typedef struct {
    int depth;
    result_t result;
#ifdef TEST
    board_t board;
#endif
} cache_t;

static double tim, time_limit;

static void init_candidate(board_t board, cboard_t candidate, int cur_id) {
    memset(candidate, 0, sizeof(cboard_t));
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (!board[i][j]) continue;
            for (int x = -1; x <= 1; x++) {
                for (int y = -1; y <= 1; y++) {
                    point_t np = (point_t){i + x, j + y};
                    if (!in_board(np) || board[np.x][np.y]) continue;
                    if (is_forbidden(board, np, cur_id, 3)) continue;
                    candidate[np.x][np.y] = 1;
                }
            }
        }
    }
    vector_t threats = vector_new(threat_t, NULL);
    threat_storage_t storage = {0};
    storage[PAT_WIN] = storage[PAT_A4] = storage[PAT_D4] = storage[PAT_A3] = storage[PAT_D3] =
        &threats;
    scan_threats(board, cur_id, storage);
    storage[PAT_D3] = NULL;
    scan_threats(board, 3 - cur_id, storage);
    for_each(threat_t, threats, threat) {
        point_t p = threat.pos;
        candidate[p.x][p.y] = 1;
    }
    vector_free(threats);
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

static result_t minimax_search(state_t state, cboard_t preset_candidate, int depth, int alpha,
                               int beta) {
    const int sgn = state.sgn, put_id = sgn == 1 ? 1 : 2;
    if (depth == 0 || state.result || get_time(tim) > time_limit)
        return (result_t){state.value, state.pos, 1};
    /*
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
        if (entry->depth > depth) {
            eval_reuse_cnt++;
            // print_emph(entry->board, entry->result.pos);
            return entry->result;
        }
    } else {
        cache_t new_cache = {depth, (result_t){0, (point_t){0, 0}}};
#ifdef TEST
        memcpy(new_cache.board, state.board, sizeof(board_t));
#endif
        entry = eval_cache_insert(eval_cache, state.hash, new_cache);
    }
    */

    result_t ret = {-EVAL_INF * sgn, {-1, -1}, 1};
    vector_t available_pos = vector_new(point_eval_t, NULL);
    if (!preset_candidate) {
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                point_t pos = (point_t){i, j};
                point_eval_t eval = {
                    .level = 0,
                    .pos = pos,
                };
                if (state.candidate[pos.x][pos.y]) {
                    vector_push_back(available_pos, eval);
                }
            }
        }
        vector_t threats = vector_new(threat_t, NULL);
        threat_storage_t storage = {0};
        storage[PAT_WIN] = storage[PAT_A4] = storage[PAT_D4] = storage[PAT_A3] = &threats;
        scan_threats(state.board, put_id, storage);
        scan_threats(state.board, 3 - put_id, storage);
        const int pat_to_level[PAT_TYPE_SIZE] = {
            [PAT_WIN] = 7, [PAT_A4] = 6, [PAT_D4] = 4, [PAT_A3] = 4, [PAT_D3] = 1, [PAT_A2] = 1,
        };
        for_each(threat_t, threats, threat) {
            for_each_ptr(point_eval_t, available_pos, eval) {
                if (point_equal(eval->pos, threat.pos)) {
                    int level = pat_to_level[threat.pattern] + (put_id == threat.id ? 0 : -1);
                    eval->level += (1 << level);
                    break;
                }
            }
        }
        vector_free(threats);
    } else {
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                point_t pos = (point_t){i, j};
                point_eval_t eval = {
                    .level = 0,
                    .pos = pos,
                };
                if (preset_candidate[i][j]) {
                    vector_push_back(available_pos, eval);
                }
            }
        }
    }
    vector_shuffle(available_pos);
    qsort(available_pos.data, available_pos.size, available_pos.element_size, eval_cmp);
    for_each(point_eval_t, available_pos, eval) {
        point_t pos = eval.pos;
        result_t child = minimax_search(put_piece(state, pos), NULL, depth - 1, alpha, beta);
        ret.tree_size += child.tree_size;
        if (sgn == 1) {  // max node
            if (child.value > alpha) {
                alpha = child.value;
                ret.pos = pos;
            }
        } else {  // min node
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
    /*
    if (entry != NULL) {
        entry->depth = depth;
        entry->result = ret;
    }
    */
    return ret;
}

result_t minimax_parallel_search(state_t init_state, cboard_t init_candidates, int depth) {
    cboard_t candidates[BOARD_AREA] = {0};
    result_t results[BOARD_AREA];
    size_t fork_cnt = 0, point_cnt = 0;
    for (int x = 0; x < BOARD_SIZE; x++) {
        for (int y = 0; y < BOARD_SIZE; y++) {
            if (init_candidates[x][y]) {
                candidates[fork_cnt][x][y] = 1;
                point_cnt++;
                // if (point_cnt % 2 == 0) {
                fork_cnt++;
                // }
            }
        }
    }
#pragma omp parallel for
    for (size_t i = 0; i < fork_cnt; i++) {
        results[i] = minimax_search(init_state, candidates[i], depth, -EVAL_INF, EVAL_INF);
    }

    double cur_time = get_time(tim);
    if (cur_time > time_limit) {
        return (result_t){0, {-1, -1}, 0};
    }
    // log("depth: %d, fork: %d, time: %.2lfms", depth, fork_cnt, cur_time);
    const int sgn = init_state.sgn;
    result_t best_result = {-EVAL_INF * sgn, {-1, -1}, 0};
    size_t tree_size = 0;
    for (size_t i = 0; i < fork_cnt; i++) {
        tree_size += results[i].tree_size;
        if (results[i].value * sgn > best_result.value * sgn) {
            best_result = results[i];
        }
    }
    best_result.tree_size = tree_size;
    return best_result;
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
    minimax_param_t param = *(minimax_param_t*)assets;
    point_t pos = trivial_move(game, param.use_vct);
    if (in_board(pos))
        return pos;
    else
        pos = initial_move(game);

    time_limit = game.time_limit * 0.95;

    cboard_t candidate = {0};
    init_candidate(game.board, candidate, game.cur_id);
    state_t state = {0};
    memcpy(state.board, game.board, sizeof(state.board));
    memcpy(state.candidate, candidate, sizeof(state.candidate));
    state.hash = zobrist_create(state.board);
    state.sgn = game.cur_id == 1 ? 1 : -1;
    state.pos = game.steps[game.count - 1];
    state.result = 0;
    state.value = eval(state.board);

    int maxdepth = 0;
    result_t best_result = {0};
    log("searching...");
    vector_t choice = vector_new(point_t, NULL);
    for (int i = 2; i < param.max_depth + 2; i += 2) {
        result_t ret;
        if (param.use_parallel)
            ret = minimax_parallel_search(state, candidate, i);
        else
            ret = minimax_search(state, NULL, i, -EVAL_INF, EVAL_INF);
        if (get_time(tim) < time_limit && in_board(ret.pos)) {
            best_result = ret, maxdepth = i, pos = best_result.pos, vector_push_back(choice, pos);
        } else
            break;
    }
    print_points(choice, PROMPT_LOG, "<");
    vector_free(choice);

    int size = best_result.tree_size;
    log("maxdepth: %d, tree size: %.2lfk, speed: %.2lf", maxdepth, size / 1000.0,
        size / get_time(tim));
    assert(in_board(pos) && available(game.board, pos));
    log("evaluate: %d", best_result.value);
    return pos;
}