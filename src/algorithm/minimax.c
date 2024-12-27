#include "minimax.h"

#include "board.h"
#include "eval.h"
#include "game.h"
#include "pattern.h"
#include "players.h"
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
    cboard_t candidate;
    zobrist_t hash;
    int value;
    int sgn;
    int result;
    point_t pos;
} state_t;

static state_t put_piece(board_t board, state_t state, point_t pos, pattern4_t pat4) {
    const int put_id = state.sgn == 1 ? 1 : 2;
    state.hash = zobrist_update(state.hash, pos, board[pos.x][pos.y], put_id);
    state.pos = pos;
    state.sgn = -state.sgn;
    if (pat4 == PAT4_WIN) {
        state.result = state.sgn;
        state.value = state.result * EVAL_MAX;
    } else {
        state.value = add_with_eval(board, state.value, pos, put_id);
    }
    state.candidate[pos.x][pos.y] = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            point_t np = (point_t){pos.x + i, pos.y + j};
            if (!in_board(np) || board[np.x][np.y]) continue;
            state.candidate[np.x][np.y] = 1;
        }
    }
    // vector_t threats = find_threats(board, pos, false);
    // for_each(threat_t, threats, threat) {
    //     point_t p = threat.pos;
    //     state.candidate[p.x][p.y] = 1;
    // }
    // vector_free(threats);
    return state;
}

static void remove_piece(board_t board, point_t pos) { board[pos.x][pos.y] = 0; }

typedef struct {
    bool valid;
    int value;
    point_t pos;
    int tree_size;
    point_t next_pos;
} result_t;

int result_serialize(char* dest, size_t size, const void* ptr) {
    result_t* result = (result_t*)ptr;
    return snprintf(dest, size, "{%c%d, val: %d}", READABLE_POS(result->pos), result->value);
}

void print_result_vector(vector_t results, const char* delim) {
    char buffer[1024];
    vector_serialize(buffer, 1024, delim, results, result_serialize);
    log_l("%s", buffer);
}

static double tim, time_limit;

static void init_candidate(board_t board, cboard_t candidate, int cur_id, int adjacent) {
    memset(candidate, 0, sizeof(cboard_t));
    const int mid = BOARD_SIZE / 2;
    for (int i = mid - adjacent; i <= mid + adjacent; i++) {
        for (int j = mid - adjacent; j <= mid + adjacent; j++) {
            point_t np = {i, j};
            if (board[np.x][np.y]) continue;
            candidate[np.x][np.y] = 1;
        }
    }
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (!board[i][j]) continue;
            for (int x = -1; x <= 1; x++) {
                for (int y = -1; y <= 1; y++) {
                    point_t np = {i + x, j + y};
                    if (candidate[np.x][np.y]) continue;
                    if (!in_board(np) || board[np.x][np.y]) continue;
                    candidate[np.x][np.y] = 1;
                }
            }
        }
    }
    vector_t threats = vector_new(threat_t, NULL);
    threat_storage_t storage = {0};
    storage[PAT_WIN] = storage[PAT_A4] = storage[PAT_D4] = storage[PAT_A3] = storage[PAT_D3] =
        &threats;
    scan_threats(board, cur_id, 0, storage);
    scan_threats(board, 3 - cur_id, 0, storage);
    for_each(threat_t, threats, threat) {
        point_t p = threat.pos;
        candidate[p.x][p.y] = 1;
    }
    vector_free(threats);
    // fboard_t tmp = {0};
    // for (int i = 0; i < BOARD_SIZE; i++) {
    //     for (int j = 0; j < BOARD_SIZE; j++) {
    //         tmp[i][j] = candidate[i][j] ? 0.3 : 0;
    //     }
    // }
    // print_prob(board, tmp);
}

typedef struct {
    int value;
    point_t pos;
} point_eval_t;

static int eval_cmp(const void* p1, const void* p2) {
    point_eval_t a = *(point_eval_t*)p1, b = *(point_eval_t*)p2;
    if (a.value != b.value) return b.value - a.value;
    return 0;
}

typedef struct {
    int max_depth;
    bool leaf_vct;
    int narrow_width;
} search_param_t;

static search_param_t search_param;

static minimax_param_t param;

typedef struct {
    int value;
    vector_t points;
    pattern_t type;
    bool is_self;
} forward_result_t;

#define STEP_INF BOARD_AREA

static void free_forward_result(forward_result_t* result) { vector_free(result->points); }

static bool array_contains(vector_t point_array, point_t pos) {
    for_each(point_t, point_array, p) {
        if (point_equal(p, pos)) return true;
    }
    return false;
}

static forward_result_t look_forward(board_t board, int self_id, vector_t* self_5,
                                     vector_t* self_a4, vector_t* self_d4, vector_t* oppo_5,
                                     vector_t* oppo_a4) {
    const int sgn = self_id == 1 ? 1 : -1, oppo_id = 3 - self_id;
    forward_result_t ret = {
        .value = 0,
        .points = {0},
        .type = PAT_EMPTY,
    };
    ret.points = vector_new(point_t, NULL);
    // log_l("s5: %d, o5: %d, sa4: %d, oa4: %d, sd4: %d", self_5.size, oppo_5.size, self_a4.size,
    // oppo_a4.size, self_d4.size);
    if (self_5->size) {
        ret.value = EVAL_MAX * sgn;
        threat_t attack = vector_get(threat_t, *self_5, 0);
        vector_push_back(ret.points, attack.pos);
        ret.type = PAT_WIN, ret.is_self = true;
    } else if (oppo_5->size) {
        bool lose = false;
        if (oppo_5->size > 1) lose = true;
        for_each(threat_t, *oppo_5, defense) {
            if (is_forbidden(board, defense.pos, self_id, 2)) {
                lose = true;
                continue;
            }
            vector_push_back(ret.points, defense.pos);
        }
        if (lose) ret.value = -EVAL_MAX * sgn;
        ret.type = PAT_WIN, ret.is_self = false;
    } else if (self_a4->size) {
        ret.value = EVAL_MAX * sgn;
        for_each(threat_t, *self_a4, attack) vector_push_back(ret.points, attack.pos);
        ret.type = PAT_A4, ret.is_self = true;
    } else if (oppo_a4->size) {
        for_each(threat_t, *oppo_a4, oppo_attack) {
            // if (!array_contains(ret.points, defense.pos) &&
            //     !is_forbidden(board, defense.pos, self_id, 2)) {
            //     vector_push_back(ret.points, defense.pos);
            // }
            vector_t defences =
                find_relative_points(DEFENSE, board, oppo_attack.pos, oppo_attack.dir.x,
                                     oppo_attack.dir.y, oppo_id, false);
            for_each(point_t, defences, p) {
                if (!array_contains(ret.points, p) && !is_forbidden(board, p, self_id, 2)) {
                    vector_push_back(ret.points, p);
                }
            }
            // print_emph_mutiple(board, defences);
            // prompt_pause();
            vector_free(defences);
            /*
            fix for
            restore_game(10000,21,(point_t[]){{7,7},{7,8},{8,6},{6,7},{8,9},{8,5},{9,9},{5,8},{7,6},{6,8},{8,8},{6,6},{6,5},{4,8},{3,8},{6,9},{6,10},{8,7},{5,10},{3,9},{5,7}});
            eval to -EVAL_MAX

            because - # o o - o is also a valid defense point, but the previous code only considers
            - o o # o -
            */
        }
        for_each(threat_t, *self_d4, attack) {
            if (!array_contains(ret.points, attack.pos)) {
                vector_push_back(ret.points, attack.pos);
            }
        }
        ret.type = PAT_A4, ret.is_self = false;
    }
    return ret;
}

// vector_t points;

static result_t minimax_search(board_t board, state_t state, cboard_t preset_candidate, int depth,
                               int alpha, int beta) {
    const int sgn = state.sgn, self_id = sgn == 1 ? 1 : 2, oppo_id = 3 - self_id;

    if (get_time(tim) > time_limit) return (result_t){false, 0, {-1, -1}, 0, {-1, -1}};
    if (state.result) {
        return (result_t){true, state.value, {-1, -1}, 1, {-1, -1}};
    }
    if (depth > search_param.max_depth) {
        return (result_t){true, state.value, {-1, -1}, 1, {-1, -1}};
    }

    result_t ret = {true, -EVAL_INF * sgn, {-1, -1}, 1, {-1, -1}};
    vector_t available_pos = {0};
    vector_t self_5, self_a4, self_d4, self_others, oppo_5, oppo_a4, oppo_others;
    self_5 = vector_new(threat_t, NULL);
    self_a4 = vector_new(threat_t, NULL);
    self_d4 = vector_new(threat_t, NULL);
    self_others = vector_new(threat_t, NULL);
    oppo_5 = vector_new(threat_t, NULL);
    oppo_a4 = vector_new(threat_t, NULL);
    oppo_others = vector_new(threat_t, NULL);
#define free_threats()                                                                         \
    vector_free(self_5), vector_free(self_a4), vector_free(self_d4), vector_free(self_others), \
        vector_free(oppo_5), vector_free(oppo_a4), vector_free(oppo_others)
    threat_storage_t self_storage = {
        [PAT_WIN] = &self_5,     [PAT_A4] = &self_a4,     [PAT_D4] = &self_d4,
        [PAT_A3] = &self_others, [PAT_D3] = &self_others, [PAT_A2] = &self_others,
    };
    threat_storage_t oppo_storage = {
        [PAT_WIN] = &oppo_5,     [PAT_A4] = &oppo_a4,     [PAT_D4] = &oppo_others,
        [PAT_A3] = &oppo_others, [PAT_D3] = &oppo_others, [PAT_A2] = &oppo_others,
    };
    scan_threats(board, self_id, self_id, self_storage);
    scan_threats(board, oppo_id, oppo_id, oppo_storage);
    if (param.optim.look_forward) {
        forward_result_t result =
            look_forward(board, self_id, &self_5, &self_a4, &self_d4, &oppo_5, &oppo_a4);
        if (result.value) {
#if DEBUG_LEVEL >= 1
            // char buffer[1024];
            // board_serialize(board, buffer);
            // FILE* f = fopen("forward.log", "a");
            // fprintf(f, "%sid: %d, sgn: %d, result: %d, eval: %d\n\n", buffer, self_id, sgn,
            //         result.value, state.value);
            // fclose(f);
#endif
            ret.value = result.value;
            if (result.points.size) {
                ret.pos = vector_get(point_t, result.points, 0);
            }
            free_forward_result(&result);
            free_threats();
            return ret;
        }
        available_pos = vector_new(point_t, NULL);
        vector_cat(available_pos, result.points);
        if (param.optim.dynamic_depth && result.type == PAT_WIN) {
            depth -= 2;  // the defend for dead 4 is trivial
            depth = max(depth, 0);
        }
        free_forward_result(&result);
    } else {
        available_pos = vector_new(point_t, NULL);
    }

    if (!available_pos.size) {
        vector_t eval_vector = vector_new(point_eval_t, NULL);
        if (!preset_candidate) {
            for (int i = 0; i < BOARD_SIZE; i++) {
                for (int j = 0; j < BOARD_SIZE; j++) {
                    if (state.candidate[i][j]) {
                        point_t pos = {i, j};
                        point_eval_t point_eval = {.value = 0, .pos = pos};
                        vector_push_back(eval_vector, point_eval);
                    }
                }
            }
            const int pat_to_level[PAT_TYPE_SIZE] = {
                [PAT_WIN] = 8, [PAT_A4] = 7, [PAT_D4] = 5, [PAT_A3] = 5, [PAT_D3] = 2, [PAT_A2] = 2,
            };
            vector_t* all_threats[] = {
                &self_5, &self_a4, &self_d4, &self_others, &oppo_5, &oppo_a4, &oppo_others,
            };
            for (size_t i = 0; i < sizeof(all_threats) / sizeof(all_threats[0]); i++) {
                vector_t threats = *all_threats[i];
                for_each(threat_t, threats, threat) {
                    for_each_ptr(point_eval_t, eval_vector, eval) {
                        if (point_equal(eval->pos, threat.pos)) {
                            int level =
                                pat_to_level[threat.pattern] + (self_id == threat.id ? 0 : -1);
                            eval->value += (1 << level);
                            break;
                        }
                    }
                }
            }
        } else {
            for (int i = 0; i < BOARD_SIZE; i++) {
                for (int j = 0; j < BOARD_SIZE; j++) {
                    if (preset_candidate[i][j]) {
                        point_eval_t eval = {
                            .value = 0,
                            .pos = {i, j},
                        };
                        vector_push_back(eval_vector, eval);
                    }
                }
            }
        }
        // vector_shuffle(eval_vector);
        qsort(eval_vector.data, eval_vector.size, eval_vector.element_size, eval_cmp);
        for_each(point_eval_t, eval_vector, eval) { vector_push_back(available_pos, eval.pos); }
        vector_free(eval_vector);
    }
    free_threats();
    if (search_param.narrow_width) {
        const size_t max_width[] = {STEP_INF, 15, 12, 10, 8, 5, 5, 5};
        const int max_width_size = sizeof(max_width) / sizeof(max_width[0]);
        available_pos.size = min(max_width[min(depth, max_width_size - 1)], available_pos.size);
    }
    for_each(point_t, available_pos, pos) {
        // vector_push_back(points, pos);
        pattern4_t pat4 = get_pattern4(board, pos, self_id, true);
        if (pat4 > PAT4_WIN) continue;
        result_t child =
            minimax_search(board, put_piece(board, state, pos, pat4), NULL, depth + 1, alpha, beta);
        remove_piece(board, pos);
        // points.size--;
        if (!child.valid) {
            ret.valid = false;
            break;
        }
        ret.tree_size += child.tree_size;
        if (sgn == 1) {  // max node
            if (child.value > alpha) {
                alpha = child.value;
                ret.pos = pos;
                ret.next_pos = child.pos;
            }
        } else {  // min node
            if (child.value < beta) {
                beta = child.value;
                ret.pos = pos;
                ret.next_pos = child.pos;
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

result_t minimax_search_entry(board_t board, state_t init_state, cboard_t init_candidates,
                              bool parallel) {
    const int self_id = init_state.sgn == 1 ? 1 : 2;
    cboard_t candidates[BOARD_AREA] = {0};
    result_t results[BOARD_AREA];
    point_t id2pos[BOARD_AREA] = {0};
    size_t fork_cnt = 0;
    for (int x = 0; x < BOARD_SIZE; x++) {
        for (int y = 0; y < BOARD_SIZE; y++) {
            if (init_candidates[x][y]) {
                point_t p = {x, y};
                if (is_forbidden(board, p, self_id, -1)) continue;
                candidates[fork_cnt][x][y] = 1;
                if (parallel) {
                    id2pos[fork_cnt] = p;
                    fork_cnt++;
                }
            }
        }
    }
    if (!parallel) fork_cnt = 1;
    if (!fork_cnt) return (result_t){false, 0, {0, 0}, 0, {0, 0}};
    // if (parallel) log_l("fork to %d threads", fork_cnt);

    board_t* boards = malloc(sizeof(board_t) * fork_cnt);
    for (size_t i = 0; i < fork_cnt; i++) {
        memcpy(boards[i], board, sizeof(board_t));
    }
#pragma omp parallel for
    for (size_t i = 0; i < fork_cnt; i++) {
        results[i] = minimax_search(boards[i], init_state, candidates[i], 0, -EVAL_INF, EVAL_INF);
    }

    for (size_t i = 0; i < fork_cnt; i++) {
        if (!results[i].valid) return (result_t){false, 0, {0, 0}, 0, {0, 0}};
    }
    // log_l("depth: %d, fork: %d, time: %.2lfms", depth, fork_cnt, cur_time);
    const int sgn = init_state.sgn;
    result_t best_result = {true, -EVAL_INF * sgn, {-1, -1}, 0, {-1, -1}};
    size_t tree_size = 0;
    for (size_t i = 0; i < fork_cnt; i++) {
        // log_l("%c%d: %d", READABLE_POS(id2pos[i]), results[i].value);
        tree_size += results[i].tree_size;
        if (results[i].value * sgn > best_result.value * sgn) {
            best_result = results[i];
        }
        if (results[i].value * sgn == -EVAL_MAX) {
            // log_l("kill %c%d", READABLE_POS(id2pos[i]));
            init_candidates[id2pos[i].x][id2pos[i].y] = 0;
        }
    }
    best_result.tree_size = tree_size;
    return best_result;
}

static point_t initial_move(game_t game) {
    const int self_id = game.cur_id, oppo_id = 3 - self_id;
    vector_t critical_threats = vector_new(threat_t, NULL);
    vector_t normal_threats = vector_new(threat_t, NULL);
    vector_t trivial_threats = vector_new(threat_t, NULL);
    threat_storage_t storage = {
        [PAT_WIN] = &critical_threats, [PAT_A4] = &critical_threats,  //
        [PAT_D4] = &normal_threats,    [PAT_A3] = &normal_threats,    //
        [PAT_D3] = &trivial_threats,   [PAT_A2] = &trivial_threats,
        [PAT_D2] = &trivial_threats,  //
    };
    scan_threats(game.board, self_id, self_id, storage);
    scan_threats(game.board, oppo_id, self_id, storage);
    vector_t candidates = vector_new(point_t, NULL);
    if (critical_threats.size) {
        // log_l("critical");
        for_each(threat_t, critical_threats, threat) { vector_push_back(candidates, threat.pos); }
    }
    if (!candidates.size && normal_threats.size) {
        // log_l("normal");
        for_each(threat_t, normal_threats, threat) { vector_push_back(candidates, threat.pos); }
    }
    if (!candidates.size) {
        point_t p = move(game, preset_players[NEURAL_NETWORK]);
        if (in_board(p)) {
            // log_l("network");
            vector_push_back(candidates, p);
        } else {
            if (trivial_threats.size) {
                // log_l("trash");
                for_each(threat_t, trivial_threats, threat) {
                    vector_push_back(candidates, threat.pos);
                }
            } else {
                // log_l("random");
                p = random_move(game), vector_push_back(candidates, p);
            }
        }
    }
    int index = rand() % candidates.size;
    point_t pos = vector_get(point_t, candidates, index);
    vector_free(critical_threats), vector_free(normal_threats), vector_free(trivial_threats);
    vector_free(candidates);
    return pos;
}

void print_candidates(board_t board, cboard_t candidates) {
    fboard_t tmp = {0};
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            tmp[i][j] = candidates[i][j] ? 0.3 : 0;
        }
    }
    print_prob(board, tmp);
}

void print_result(result_t result, double duration) {
    log_l("depth %d%s, pos %c%d, %c%d, size %.2lfk, time %.2lfms, value %d, speed %.2lf",
          search_param.max_depth, search_param.narrow_width ? "" : "(full)",
          READABLE_POS(result.pos), READABLE_POS(result.next_pos), result.tree_size * 1e-3,
          duration, result.value, result.tree_size / duration);
}

point_t minimax(game_t game, const void* assets) {
    // if (!points.data) points = vector_new(point_t, NULL);
    if (game.count == 0) {
        return (point_t){BOARD_SIZE / 2, BOARD_SIZE / 2};
    }
    tim = record_time();
    param = *(minimax_param_t*)assets;
    point_t pos = trivial_move(game, min(1500, game.time_limit / 2), true, param.optim.begin_vct);
    if (in_board(pos)) {
        return pos;
    } else {
        pos = initial_move(game);
        log_l("initial pos: %c%d", READABLE_POS(pos));
    }

    time_limit = game.time_limit * 0.98;

    cboard_t candidate = {0};
    init_candidate(game.board, candidate, game.cur_id, param.strategy.adjacent);
    state_t state = {0};
    memcpy(state.candidate, candidate, sizeof(state.candidate));
    state.sgn = game.cur_id == 1 ? 1 : -1;
    state.pos = game.steps[game.count - 1];
    state.result = 0;
    state.value = eval(game.board);

    int calculated_depth = 0;
    result_t best_result = {0};
    // log_l("searching...");
    vector_t preset_params = vector_new(search_param_t, NULL);
    for (int i = 2; i < param.max_depth + 2; i += 2) {
        search_param_t p = {i, false, param.optim.narrow_width};
        vector_push_back(preset_params, p);
        // if (param.optim.leaf_vct_depth && i >= param.optim.leaf_vct_depth) {
        //     p.leaf_vct = true;
        //     vector_push_back(preset_params, p);
        //     log_l("vct on depth %d", i);
        // }
    }
    vector_t choice = vector_new(point_t, NULL);
    for_each(search_param_t, preset_params, preset_param) {
        search_param = preset_param;
        result_t ret = {0};
        double start_time;
    start:
        start_time = record_time();
        ret = minimax_search_entry(game.board, state, candidate, param.parallel);
        if (ret.valid == 0) break;
        print_result(ret, get_time(start_time));
        calculated_depth = preset_param.max_depth, vector_push_back(choice, ret.pos);
        if (ret.value * state.sgn != -EVAL_MAX) {
            best_result = ret, pos = ret.pos;
        }
        if (ret.value == EVAL_MAX || ret.value == -EVAL_MAX) {
            if (!search_param.narrow_width) {
                break;
            } else {
                search_param.narrow_width = false;
                goto start;  // if find win or lose in narrowed width, then check it in full width
            }
        }
    }
    vector_free(choice);
    vector_free(preset_params);

    int size = best_result.tree_size;
    log_l("maxdepth: %d, tree size: %.2lfk, speed: %.2lf", calculated_depth, size / 1000.0,
          size / get_time(tim));
    assert(in_board(pos) && available(game.board, pos));
    log_l("evaluate: %d", best_result.value);
    return pos;
}