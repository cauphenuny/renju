#include "minimax.h"

#include "board.h"
#include "eval.h"
#include "game.h"
#include "pattern.h"
#include "trivial.h"
#include "util.h"
#include "vct.h"
#include "vector.h"
#include "zobrist.h"

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EVAL_MAX (EVAL_INF - 1)

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
    if (state.result) state.value = state.result * EVAL_MAX;
    state.sgn = -state.sgn;
    state.candidate[pos.x][pos.y] = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            point_t np = (point_t){pos.x + i, pos.y + j};
            if (!in_board(np) || state.board[np.x][np.y]) continue;
            if (is_forbidden(state.board, np, 3 - put_id, 2)) continue;
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
    bool valid;
    int value;
    point_t pos;
    int tree_size;
} result_t;

void result_serialize(char* dest, const void* ptr) {
    result_t* result = (result_t*)ptr;
    snprintf(dest, 32, "{%c%d, val: %d}", READABLE_POS(result->pos), result->value);
}

void print_result_vector(vector_t results, const char* delim) {
    char buffer[1024];
    vector_serialize(buffer, delim, results, result_serialize);
    log("%s", buffer);
}

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
        if (is_forbidden(board, p, cur_id, 3)) continue;
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

static int max_depth;

static minimax_param_t param;

typedef struct {
    int value;
    vector_t points;
} forward_result_t;

static void free_forward_result(forward_result_t* result) { vector_free(result->points); }

static forward_result_t look_forward(board_t board, int cur_id) {
    const int sgn = cur_id == 1 ? 1 : -1;
    forward_result_t ret = {
        .value = 0,
        .points = vector_new(point_t, NULL),
    };
    vector_t self_5 = vector_new(threat_t, NULL);
    vector_t self_a4 = vector_new(threat_t, NULL);
    vector_t self_d4 = vector_new(threat_t, NULL);
    scan_threats(board, cur_id,
                 (threat_storage_t){[PAT_WIN] = &self_5, [PAT_A4] = &self_a4, [PAT_D4] = &self_d4});
    vector_t oppo_5 = vector_new(threat_t, NULL);
    vector_t oppo_a4 = vector_new(threat_t, NULL);
    scan_threats(board, 3 - cur_id, (threat_storage_t){[PAT_WIN] = &oppo_5, [PAT_A4] = &oppo_a4});
    if (self_5.size) {
        ret.value = EVAL_MAX * sgn;
        threat_t attack = vector_get(threat_t, self_5, 0);
        vector_push_back(ret.points, attack.pos);
    } else if (oppo_5.size) {
        bool lose = false;
        if (oppo_5.size > 1) lose = true;
        for_each(threat_t, oppo_5, defense) {
            if (is_forbidden(board, defense.pos, cur_id, 2)) {
                lose = true;
                continue;
            }
            vector_push_back(ret.points, defense.pos);
        }
        if (lose) ret.value = -EVAL_MAX * sgn;
    } else if (self_a4.size) {
        ret.value = EVAL_MAX * sgn;
        for_each(threat_t, self_a4, attack) vector_push_back(ret.points, attack.pos);
    } else if (oppo_a4.size) {
        for_each(threat_t, oppo_a4, defense) {
            if (is_forbidden(board, defense.pos, cur_id, 2)) {
                continue;
            }
            vector_push_back(ret.points, defense.pos);
        }
        for_each(threat_t, self_d4, attack) vector_push_back(ret.points, attack.pos);
    }
    vector_free(self_5), vector_free(self_a4), vector_free(self_d4);
    vector_free(oppo_5), vector_free(oppo_a4);
    return ret;
}

static result_t minimax_search(state_t state, cboard_t preset_candidate, int depth, int alpha,
                               int beta) {
    const int sgn = state.sgn, cur_id = sgn == 1 ? 1 : 2;

    if (get_time(tim) > time_limit) return (result_t){false, 0, {-1, -1}, 0};
    if (depth > max_depth || state.result) return (result_t){true, state.value, state.pos, 1};

    result_t ret = {true, -EVAL_INF * sgn, {-1, -1}, 1};
    vector_t available_pos = {0};
    if (param.optim.look_forward) {
        forward_result_t result = look_forward(state.board, cur_id);
        if (result.value) {
#if DEBUG_LEVEL >= 1
            // char buffer[1024];
            // board_serialize(state.board, buffer);
            // FILE* f = fopen("forward.log", "a");
            // fprintf(f, "%sid: %d, sgn: %d, result: %d, eval: %d\n\n", buffer, cur_id, sgn,
            //         result.value, state.value);
            // fclose(f);
#endif
            ret.value = result.value;
            if (result.points.size) {
                ret.pos = vector_get(point_t, result.points, 0);
            }
            free_forward_result(&result);
            return ret;
        }
        available_pos = vector_new(point_t, NULL);
        vector_cat(available_pos, result.points);
        free_forward_result(&result);
    } else {
        available_pos = vector_new(point_t, NULL);
    }

    if (param.optim.search_vct && (max_depth - depth > 4)) {
        double vct_time = 0.5;
        if (depth < 5) vct_time += (5 - depth) * 1;
        vector_t vct_sequence = vct(false, state.board, cur_id, vct_time);
        // FILE* f = fopen("vct.csv", "a");
        // fprintf(f, "%d,%lu,\n", state.value * sgn, vct_sequence.size);
        // fclose(f);
        if (vct_sequence.size) {
            point_t pos = vector_get(point_t, vct_sequence, 0);
            vector_free(vct_sequence);
            vector_free(available_pos);
            return (result_t){true, EVAL_MAX * sgn, pos, 1};
        } else {
            vector_free(vct_sequence);
        }
    }

    if (!available_pos.size) {
        vector_t eval_vector = vector_new(point_eval_t, NULL);
        if (!preset_candidate) {
            for (int i = 0; i < BOARD_SIZE; i++) {
                for (int j = 0; j < BOARD_SIZE; j++) {
                    point_t pos = (point_t){i, j};
                    point_eval_t eval = {.level = 0, .pos = pos};
                    if (state.candidate[pos.x][pos.y]) {
                        vector_push_back(eval_vector, eval);
                    }
                }
            }
            vector_t threats = vector_new(threat_t, NULL);
            threat_storage_t storage = {0};
            storage[PAT_WIN] = storage[PAT_A4] = storage[PAT_D4] = storage[PAT_A3] = &threats;
            scan_threats(state.board, cur_id, storage);
            scan_threats(state.board, 3 - cur_id, storage);
            const int pat_to_level[PAT_TYPE_SIZE] = {
                [PAT_WIN] = 7, [PAT_A4] = 6, [PAT_D4] = 4, [PAT_A3] = 4, [PAT_D3] = 1, [PAT_A2] = 1,
            };
            for_each(threat_t, threats, threat) {
                for_each_ptr(point_eval_t, eval_vector, eval) {
                    if (point_equal(eval->pos, threat.pos)) {
                        int level = pat_to_level[threat.pattern] + (cur_id == threat.id ? 0 : -1);
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
                        vector_push_back(eval_vector, eval);
                    }
                }
            }
        }
        vector_shuffle(eval_vector);
        qsort(eval_vector.data, eval_vector.size, eval_vector.element_size, eval_cmp);
        for_each(point_eval_t, eval_vector, eval) { vector_push_back(available_pos, eval.pos); }
        vector_free(eval_vector);
    }
    for_each(point_t, available_pos, pos) {
        result_t child = minimax_search(put_piece(state, pos), NULL, depth + 1, alpha, beta);
        if (!child.valid) {
            ret.valid = false;
            break;
        }
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
    return ret;
}

result_t minimax_parallel_search(state_t init_state, cboard_t init_candidates) {
    cboard_t candidates[BOARD_AREA] = {0};
    result_t results[BOARD_AREA];
    size_t fork_cnt = 0;
    for (int x = 0; x < BOARD_SIZE; x++) {
        for (int y = 0; y < BOARD_SIZE; y++) {
            if (init_candidates[x][y]) {
                candidates[fork_cnt][x][y] = 1;
                fork_cnt++;
            }
        }
    }
#pragma omp parallel for
    for (size_t i = 0; i < fork_cnt; i++) {
        results[i] = minimax_search(init_state, candidates[i], 0, -EVAL_INF, EVAL_INF);
    }

    for (size_t i = 0; i < fork_cnt; i++) {
        if (!results[i].valid) return (result_t){false, 0, {0, 0}, 0};
    }
    // log("depth: %d, fork: %d, time: %.2lfms", depth, fork_cnt, cur_time);
    const int sgn = init_state.sgn;
    result_t best_result = {true, -EVAL_INF * sgn, {-1, -1}, 0};
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
        [PAT_WIN] = &better_move, [PAT_A4] = &better_move,                     //
        [PAT_D4] = &normal_move,  [PAT_A3] = &normal_move,                     //
        [PAT_D3] = &trash,        [PAT_A2] = &trash,       [PAT_D2] = &trash,  //
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
    if (game.count == 0) {
        return (point_t){BOARD_SIZE / 2, BOARD_SIZE / 2};
    }
    tim = record_time();
    param = *(minimax_param_t*)assets;
    point_t pos =
        trivial_move(game.board, game.cur_id, game.time_limit / 20.0, param.optim.begin_vct);
    if (in_board(pos))
        return pos;
    else
        pos = initial_move(game);
    log("initial pos: %c%d", READABLE_POS(pos));

    time_limit = game.time_limit * 0.95;

    cboard_t candidate = {0};
    init_candidate(game.board, candidate, game.cur_id);
    state_t state = {0};
    memcpy(state.board, game.board, sizeof(state.board));
    memcpy(state.candidate, candidate, sizeof(state.candidate));
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
        if (param.parallel)
            max_depth = i, ret = minimax_parallel_search(state, candidate);
        else
            max_depth = i, ret = minimax_search(state, NULL, 0, -EVAL_INF, EVAL_INF);
        if (ret.valid == 0) break;
        log("depth %d, pos %c%d, time %.2lfms, value %d", i, READABLE_POS(ret.pos), get_time(tim),
            ret.value);
        maxdepth = i, vector_push_back(choice, ret.pos);
        if (ret.value * state.sgn != -EVAL_MAX) {
            best_result = ret, pos = ret.pos;
        }
    }
    vector_free(choice);

    int size = best_result.tree_size;
    log("maxdepth: %d, tree size: %.2lfk, speed: %.2lf", maxdepth, size / 1000.0,
        size / get_time(tim));
    assert(in_board(pos) && available(game.board, pos));
    log("evaluate: %d", best_result.value);
    return pos;
}