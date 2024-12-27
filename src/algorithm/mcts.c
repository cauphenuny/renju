#include "mcts.h"

#include "board.h"
#include "distribute.h"
#include "eval.h"
#include "network.h"
#include "neuro.h"
#include "pattern.h"
#include "trivial.h"
#include "util.h"

#include <assert.h>
#undef log
#include <math.h>
#define log log_l
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define WINPOS_SIZE 2

typedef struct {
    comp_board_t board, visited;
    int piece_cnt, visited_cnt;
    int count;
    double prior_P, post_Q;
    bool evaluated;
    int capacity;
    point_t begin, end;
    point_t pos;
    point_t win_pos[WINPOS_SIZE];
    int8_t id;
    int8_t score;
} state_t;

typedef struct edge_t {
    struct node_t* to;
    struct edge_t* next;
} edge_t;

typedef struct node_t {
    state_t state;
    int child_cnt;
    struct edge_t* child_edge;
    struct node_t* parent;
} node_t;

/// @brief memory buffer for node and edge, prevent from frequent malloc
static node_t* node_buffer;
static edge_t* edge_buffer;

static int tot, edge_tot;
static int current_check_depth;
static int predict_cnt;
static double start_time;
static double predict_sum_time, time_limit;
static mcts_param_t param;
static double prior_weight;

#define MAX_TREE_SIZE 30001000
#define NODE_LIMIT    30000000

static int max_tree_size = MAX_TREE_SIZE, node_limit = NODE_LIMIT;

/*
/// @brief print data of state {st}
static void print_state(const state_t st)
{
    log("print state:");
    log("board:");
    print_compressed_board(st.board, st.pos);
    log("visited:");
    print_compressed_board(st.visited, st.pos);
    log("id: %d, score: %d", st.id, st.score);
    log("result: %d, count: %d", st.result, st.count);
    log("begin: (%d, %d), end: (%d, %d), capacity: %d", st.begin.x, st.begin.y, st.end.x, st.end.y,
        st.capacity);
    log("piece_cnt: %d, visited_cnt: %d", st.piece_cnt, st.visited_cnt);
    log("pos: (%d, %d)", st.pos.x, st.pos.y);
    for (int i = 0; i < 2; i++) {
        log("win pos #%d: (%d, %d)", i, st.win_pos[i].x, st.win_pos[i].y);
    }
    log("done");
}
*/

/// @brief get positions which are winning pos after last move
static void get_win_pos(state_t* st) {
    const point_t pos = st->pos;
    memset(st->win_pos, -1, sizeof(st->win_pos));
    if (!in_board(pos)) return;
    const line_t* board = st->board;
    const int id = st->id, oppo = 3 - st->id;
    if (!id) {
        return;
    }
    point_t tmp;
    // print_state(*st), log("start");
    int8_t dx, dy;
    const int mid = SEGMENT_LEN / 2;  // . . . . . X . . . . .
    for (int i = 0, val, pattern, piece, cnt = 0; i < 4 && cnt < WINPOS_SIZE; i++) {
        dx = DIRS[i][0], dy = DIRS[i][1];
        val = 0;
        for (int j = mid; j >= -mid; j--) {
            tmp = (point_t){pos.x + dx * j, pos.y + dy * j};
            piece = in_board(tmp) ? get(board, tmp) : oppo;
            val = val * PIECE_SIZE +
                  (piece == id ? SELF_PIECE : (piece == oppo ? OPPO_PIECE : EMPTY_PIECE));
        }
        pattern = to_pattern(val, id == 1);
        if (pattern == PAT_D4 || pattern == PAT_A4) {
            int col[2];
            get_attack_columns(val, id == 1, col, 2);
            for (int j = 0; j < 2 && cnt < WINPOS_SIZE; j++) {
                if (col[j] != -1) {
                    tmp = (point_t){pos.x + dx * (col[j] - mid), pos.y + dy * (col[j] - mid)};
                    assert(in_board(tmp) && !get(board, tmp));
                    if (!is_forbidden_comp(st->board, tmp, id, current_check_depth)) {
                        st->win_pos[cnt++] = tmp;
                    }
                }
            }
        }
    }
}

/// @brief create state from given info
/// @param begin left-top corner of the area of board
/// @param end right-bottom corner of the area of board
static state_t empty_state(point_t begin, point_t end, int next_id) {
    state_t st = {0};
    st.begin = begin;
    st.end = end;
    st.id = next_id;
    st.prior_P = 1;
    st.capacity = ((int)end.x - begin.x) * ((int)end.y - begin.y);
    st.pos = (point_t){-1, -1};
    get_win_pos(&st);
    return st;
}

/// @brief update {state} with new piece at {pos}
/// @param new_pat4 pre-calculated pattern
/// @return updated state
static state_t update_state(state_t state, point_t pos, pattern4_t new_pat4) {
    add(state.board, pos, state.id);
    memcpy(state.visited, state.board, sizeof(state.board));
    state.pos = pos;
    if (in_area(pos, state.begin, state.end)) {
        state.piece_cnt++;
    }
    // state.result = 0
    state.count = 0, state.post_Q = 0;
    state.visited_cnt = state.piece_cnt;
    assert(!state.score);
    if (new_pat4 == PAT4_WIN) {
        state.score = (state.id == 1) ? 1 : -1;  // current player win
        // log("win situation %d: ", state.score);
        // print_compressed_board(state.board, pos);
        // prompt_pause();
    }
    get_win_pos(&state);
    state.id = 3 - state.id;  // change player
    state.prior_P = 1, state.evaluated = false;
    return state;
}

/// @brief create a node containing {state}
/// @param state state of the node
static node_t* create_node(state_t state) {
    if (tot >= node_limit) return NULL;
    node_t* node = node_buffer + tot++;
    // node = (node_t*)malloc(sizeof(node_t));
    memset(node, 0, sizeof(node_t));
    memcpy(&node->state, &state, sizeof(state_t));
    return node;
}

/// @brief set {child} to be one of the children of {node}
/// @return the count of {node}'s children
static int append_child(node_t* node, node_t* child) {
    edge_t* child_edge = edge_buffer + edge_tot++;
    // edge_t* child_edge = (edge_t*)malloc(sizeof(edge_t));
    memset(child_edge, 0, sizeof(edge_t));

    child_edge->next = node->child_edge;
    child_edge->to = child;
    node->child_edge = child_edge;
    node->child_cnt++;

    child->parent = node;
    return node->child_cnt;
}

#undef log
/// @brief get the evaluation of a node by ucb formula
static double ucb_eval(const node_t* node) {
    const int flag = (node->parent->state.id == 1) ? 1 : -1;
    // const int win_cnt = flag * node->state.result;
    // const double post_Q = (double)win_cnt / node->state.count;
    const double post_Q = flag * node->state.post_Q;
    // const double Q = node->state.evaluated
    //                      ? (post_Q * (1 - prior_weight) + node->state.prior_eval * prior_weight)
    //                      : post_Q;
    const double U = sqrt(log(node->parent->state.count) / node->state.count);
    const double P = pow(node->state.prior_P, 0.5);
    // const double P = node->state.prior_P;
    return post_Q + param.C_puct * P * U;
}

static double node_entropy(const node_t* node) {
    double sum = 0;
    for (const edge_t* e = node->child_edge; e; e = e->next) {
        const double prob = (double)e->to->state.count / node->state.count;
        sum += prob * log(1 / prob);
    }
    return sum;
}
#define log log_l

/// @brief select best child of {node} by visiting count, the node must have at least one child
static node_t* max_count(node_t* node) {
    if (!node->child_edge) return node;
    const edge_t* cur = node->child_edge;
    node_t* child = cur->to;
    while (cur != NULL) {
        if (cur->to->state.count > child->state.count) {
            child = cur->to;
        }
        cur = cur->next;
    }
    return child;
}

/// @brief select child of {node} which has the minimum count, the node must have at least one child
static node_t* min_count(node_t* node) {
    if (!node->child_edge) return node;
    const edge_t* cur = node->child_edge;
    node_t* child = cur->to;
    while (cur != NULL) {
        if (cur->to->state.count < child->state.count) {
            child = cur->to;
        }
        cur = cur->next;
    }
    return child;
}

/// @brief select best child of {node} by ucb value, the node must have at least one child
static node_t* ucb_select(const node_t* node) {
    assert(node->child_edge);
    const edge_t* cur = node->child_edge;
    node_t* best = cur->to;
    while (cur != NULL) {
        if (ucb_eval(cur->to) > ucb_eval(best)) {
            best = cur->to;
        }
        cur = cur->next;
    }
    return best;
}

static void backpropagate(node_t* node, double value, int count, bool remain_count);

static void evaluate_children(node_t* node) {
    if (predict_sum_time > time_limit / 3) return;
    double time = record_time();
    board_t board;
    decode(node->state.board, board);
    prediction_t prediction = predict(param.network, board, node->state.pos, node->state.id);
    // log("evaluated:");
    // print_prob(board, prediction.prob);
    for (edge_t* e = node->child_edge; e; e = e->next) {
        const point_t pos = e->to->state.pos;
        e->to->state.prior_P = (double)prediction.prob[pos.x][pos.y] * BOARD_AREA;
        // for normalization, now E(prob) = 1
    }
    // prompt_pause();
    node->state.evaluated = true;
    predict_sum_time += get_time(time);
    predict_cnt++;
}

static void trivial_evaluate_children(node_t* node) {
    if (predict_sum_time > time_limit / 3) return;
    double time = record_time();

    vector_t threats[5];
    for (int i = 0; i < 5; i++) threats[i] = vector_new(threat_t, NULL);
    threat_storage_t storage = {
        [PAT_WIN] = &threats[0],                          //
        [PAT_A4] = &threats[1],  [PAT_D4] = &threats[2],  //
        [PAT_A3] = &threats[3],  [PAT_D3] = &threats[4],  //
    };
    board_t board;
    decode(node->state.board, board);
    scan_threats(board, node->state.id, node->state.id, storage);
    scan_threats(board, 3 - node->state.id, node->state.id, storage);
    float prob[BOARD_AREA] = {0};
    float weight[5] = {3, 1.5, 1, 0.5, 0.3};
    for (int i = 0; i < 5; i++) {
        for_each(threat_t, threats[i], threat) {
            point_t pos = threat.pos;
            prob[pos.x * BOARD_SIZE + pos.y] += weight[i];
        }
    }
    for (int i = 0; i < 5; i++) vector_free(threats[i]);

    softmax_array(prob, BOARD_AREA);
    for (edge_t* e = node->child_edge; e; e = e->next) {
        const point_t pos = e->to->state.pos;
        e->to->state.prior_P = prob[pos.x * BOARD_SIZE + pos.y] * BOARD_AREA;
    }
    predict_sum_time += get_time(time);
    predict_cnt++;
    node->state.evaluated = true;
}

/// @brief check if state is terminated
static bool terminated(state_t st) {
    if (st.score || st.piece_cnt == st.capacity) return true;
    return false;
}

/// @brief put a piece at {pos} on board
/// @return node pointer containing new state
static node_t* put_piece(node_t* node, point_t pos, pattern4_t new_pat4) {
    add(node->state.visited, pos, 1);
    if (pos.x >= node->state.begin.x && pos.x < node->state.end.x &&  //
        pos.y >= node->state.begin.y && pos.y < node->state.end.y) {
        node->state.visited_cnt++;
    }

    node_t* child = create_node(update_state(node->state, pos, new_pat4));
    if (child != NULL) append_child(node, child);
    return child;
}

/// @brief find a child of {node} which is at {pos}, if no such child, then create one
/// @return child at pos
static node_t* find_child(node_t* node, point_t pos) {
    // log("id: %d", parent->state.id);
    // prompt_pause();
    for (const edge_t* edge = node->child_edge; edge != NULL; edge = edge->next) {
        if (edge->to->state.pos.x == pos.x && edge->to->state.pos.y == pos.y) {
            return edge->to;
        }
    }
    return put_piece(
        node, pos,
        virtual_pat4type_comp(node->state.board, pos, node->state.id, current_check_depth));
}

/// @brief traverse the tree to find a leaf node
static node_t* traverse(node_t* node) {
    if (node == NULL || terminated(node->state)) {
        return node;
    }
    if (current_check_depth > param.check_depth) current_check_depth--;
    state_t state = node->state;
    const int id = state.id;
    const node_t* parent = node->parent;
    for (int i = 0; i < 2; i++) {
        const point_t win_pos = parent->state.win_pos[i];
        if (in_board(win_pos) && get(state.board, win_pos) == 0) {
            assert(!is_forbidden_comp(state.board, win_pos, id, current_check_depth));
            return traverse(find_child(node, win_pos));
        }
    }
    for (int i = 0; i < 2; i++) {
        const point_t danger_pos = state.win_pos[i];
        if (in_board(danger_pos)) {
            if (!is_forbidden_comp(state.board, danger_pos, id, current_check_depth)) {
                return traverse(find_child(node, danger_pos));
            }
        }
    }
    point_t empty_pos = {-1, -1};
    int res;
    pattern4_t pat4 = PAT4_OTHERS;
    while ((res = state.capacity - state.visited_cnt) > 0 && !in_board(empty_pos)) {
        empty_pos = (point_t){-1, -1};
        const int index = (rand() % res) + 1;
        int8_t i = state.begin.x, j = state.begin.y;
        for (int t = 0, cnt = 0; t < state.capacity; t++, j++) {
            if (j >= state.end.y) i++, j = state.begin.y;
            const point_t pos = (point_t){i, j};
            if (!get(state.visited, pos)) cnt++;
            if (cnt == index) {
                // log("res = %d, pos = %d, choose %d, %d", res, pos, i, j);
                pat4 = virtual_pat4type_comp(state.board, pos, id, current_check_depth);
                if (pat4 > PAT4_WIN) {
                    state.visited_cnt++;
                    add(state.visited, pos, 1);
                } else {
                    empty_pos = pos;
                }
                break;
            }
        }
    }
    node->state = state;
    if (in_board(empty_pos)) {
        return traverse(put_piece(node, empty_pos, pat4));
    }
    if (node->child_cnt) {
        if (!node->state.evaluated) {
            switch (param.eval_type) {
                case EVAL_NETWORK: evaluate_children(node); break;
                case EVAL_HEURISTIC: trivial_evaluate_children(node); break;
                default: break;
            }
        }
        return traverse(ucb_select(node));
    }
    node->state.score = id == 1 ? -1 : 1;  // no available space
    return node;
}

static void backpropagate(node_t* node, double value, int count, bool remain_count) {
    while (node != NULL) {
        if (remain_count) {
            node->state.post_Q += value * count / node->state.count;
        } else {
            node->state.count += count;
            node->state.post_Q += (value - node->state.post_Q) * count / node->state.count;
        }
        node = node->parent;
    }
}

static void simulate(node_t* start_node) {
#ifndef NO_FORBID
    current_check_depth = 4;
#else
    current_check_depth = 0;
#endif
    node_t* leaf = traverse(start_node);
    if (leaf == NULL) return;
    backpropagate(leaf, leaf->state.score, 1, false);
    // const int score = leaf->state.score;
    // while (leaf != NULL) {
    //     leaf->state.result += score;
    //     leaf->state.count++;
    //     assert(leaf->state.result <= leaf->state.count && -leaf->state.result <=
    //     leaf->state.count); leaf = leaf->parent;
    // }
}

/// @brief get next move by mcts algorithm
point_t mcts(game_t game, const void* assets) {
    if (game.count == 0) {
        return (point_t){BOARD_SIZE / 2, BOARD_SIZE / 2};
    }
    start_time = record_time();
    param = *((mcts_param_t*)assets);

    point_t trivial_pos =
        trivial_move(game, min(1000, game.time_limit / 5.0), false, param.use_vct && !param.is_train);
    if (in_board(trivial_pos)) {
        if (param.output_prob) param.output_prob[trivial_pos.x][trivial_pos.y] = 1;
        print_prob(game.board, param.output_prob);
        return trivial_pos;
    }

    while (node_buffer == NULL || edge_buffer == NULL) {
        free(node_buffer), node_buffer = NULL;
        free(edge_buffer), edge_buffer = NULL;
        node_buffer = (node_t*)malloc(max_tree_size * sizeof(node_t));
        edge_buffer = (edge_t*)malloc(max_tree_size * sizeof(edge_t));
        if (edge_buffer == NULL || edge_buffer == NULL) max_tree_size /= 2, node_limit /= 2;
    }

    srand(time(0));

    tot = edge_tot = 0;
    time_limit = (double)max(game.time_limit - 200, game.time_limit * 0.9);

    point_t wrap_begin, wrap_end;
    int radius = param.wrap_rad;
    do {
        wrap_area(game.board, &wrap_begin, &wrap_end, radius);
    } while (((wrap_end.y - wrap_begin.y) * (wrap_end.x - wrap_begin.x) < 80) && ++radius);

    node_t* root = create_node(empty_state(wrap_begin, wrap_end, 1));
    for (int i = 0; i < game.count; i++) {
        root = put_piece(root, game.steps[i], PAT4_OTHERS);
    }
#if DEBUG_LEVEL > 0
    for (int i = 0; i < WINPOS_SIZE; i++) {
        const point_t pos = root->parent->state.win_pos[i];
        if (in_board(pos)) {
            log("(%d, %d)", pos.x, pos.y);
        }
    }
#endif

    predict_sum_time = predict_cnt = 0;

    int cnt = 0;
    double tim;
    const int target_count = param.min_count * (root->state.capacity + game.count * 2);

    log("simulating (C: %.3lf, time: %d~%d, check: %d)", param.C_puct, param.min_time,
        game.time_limit, param.check_depth);
    while ((tim = get_time(start_time)) < param.min_time ||
           max_count(root)->state.count < target_count) {
        if (tim > time_limit || tot >= node_limit) break;
        // prior_weight = 1 - ((double)tim / time_limit) / 2;
        prior_weight = 1;
        simulate(root), cnt++;
        // if (!root->state.evaluated && root->state.visited_cnt == root->state.capacity) {
        //     if (param.network) {
        //         evaluate_children(root);
        //     } else {
        //         log("evaluated root");
        //         trivial_evaluate_children(root);
        //     }
        // }
    }
    log("simulated %d times, average %.2lf us, speed %.2lf", cnt, tim * 1000 / cnt, cnt / tim);
    log("consumption: %.2lf ms", tim);
    log("  - search(%.1lf%%): size of search tree: %d", (tim - predict_sum_time) * 100 / tim, tot);
    if (predict_cnt) {
        log("  - evaluate(%.1lf%%): evaluated %d nodes, average %.3lf ms",
            predict_sum_time * 100 / tim, predict_cnt, predict_sum_time / predict_cnt);
    }
    log("node_entropy: %.3lf, min count: %d, max count: %d", node_entropy(root),
        min_count(root)->state.count, max_count(root)->state.count);
    // log("probability result:");
    fboard_t prob = {0};
    for (const edge_t* e = root->child_edge; e; e = e->next) {
        const node_t* child = e->to;
        prob[child->state.pos.x][child->state.pos.y] =
            (double)child->state.count / root->state.count;
    }
    if (param.output_prob != NULL) {
        memcpy(param.output_prob, prob, sizeof(prob));
    }
    node_t* move;

    if (param.network && param.is_train) {
        float error[BOARD_AREA], dist[BOARD_AREA] = {0};
        simple_dirichlet_distribution(0.3, error, BOARD_AREA);
        for (const edge_t* e = root->child_edge; e; e = e->next) {
            const node_t* child = e->to;
            dist[child->state.pos.x * BOARD_SIZE + child->state.pos.y] = child->state.count;
        }
        dist_log(dist, BOARD_AREA);
        dist_set_temperature(dist, BOARD_AREA, 1);
        dist_softmax(dist, BOARD_AREA);
        log("train mode, add dirichlet error");
        for (int i = 0; i < BOARD_AREA; i++) dist[i] = 0.75 * dist[i] + 0.25 * error[i];
        float max_prob = 0;
        for (int i = 0; i < BOARD_AREA; i++) max_prob = max(max_prob, dist[i]);
        log("max prob: %.3f", max_prob);
        print_prob(game.board, (pfboard_t)dist);
        point_t pos = {-1, -1};
        while (!in_board(pos)) {
            int index = sample_from_dist(dist, BOARD_AREA);
            log("selected prob: %f", dist[index]);
            point_t p = (point_t){index / BOARD_SIZE, index % BOARD_SIZE};
            if (!game.board[p.x][p.y] && !is_forbidden(game.board, p, game.cur_id, 3)) {
                pos = p;
            }
        }
        move = find_child(root, pos);
    } else {
        move = max_count(root);
    }

    const state_t st = move->state;
    int level, f = game.cur_id == 1 ? 1 : -1;
    const double rate = ((f * st.post_Q + 1) / 2 * 100);
    if (rate > 15 && rate < 85)
        level = PROMPT_LOG;
    else
        level = PROMPT_WARN;
    log_add(level, "(%d, %d) -> prob: %.2lf%% (%d), win: %.2lf%%.", st.pos.x, st.pos.y,
            (double)st.count / root->state.count * 100, st.count, rate);
    return st.pos;
}

point_t mcts_nn(const game_t game, const void* assets) {
    param = *((mcts_param_t*)assets);
    if (!param.network) {
        log_e("network not found, press enter to continue");
        prompt_pause();
        return (point_t){GAMECTRL_GIVEUP, 1};
    }
    return mcts(game, assets);
}

#ifdef TEST
#include "src/_tests/mcts.txt"
#endif
