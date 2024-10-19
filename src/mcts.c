// author: Cauphenuny
// date: 2024/07/24
#include "mcts.h"

#include "board.h"
#include "util.h"

#include <assert.h>
#undef log
#include <math.h>
#define log log_l
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if BOARD_SIZE <= 16
typedef uint32_t line_t;
#elif BOARD_SIZE <= 32
typedef uint64_t line_t;
#else
#    error "board size too large!"
#endif

typedef line_t cprboard_t[BOARD_SIZE];

#define WINPOS_SIZE 2

typedef struct {
    cprboard_t board, visited;
    int piece_cnt, visited_cnt;
    int result, count;
    int capacity;
    point_t begin, end;
    point_t pos;
    point_t win_pos[WINPOS_SIZE];
    int8_t id;
    int8_t score;
} status_t;

typedef struct edge_t {
    struct node_t* to;
    struct edge_t* next;
} edge_t;

typedef struct node_t {
    status_t status;
    int child_cnt;
    struct edge_t* child_edge;
    struct node_t* parent;
} node_t;

/// @brief memory buffer for node and edge, prevent from frequent malloc
static node_t* node_buffer;
static edge_t* edge_buffer;

static int tot, edge_tot;
static int first_id;
static mcts_param_t param;

/// @brief encode from raw board to compressed board
static void encode(const board_t src, cprboard_t dest)
{
    for (int i = 0; i < BOARD_SIZE; i++) {
        dest[i] = 0;
        for (int j = BOARD_SIZE - 1; j >= 0; j--) {
            dest[i] = dest[i] * 4 + src[i][j];
        }
    }
}

/// @brief decode from compressed board to raw board
static void decode(const cprboard_t src, board_t dest)
{
    for (int i = 0; i < BOARD_SIZE; i++) {
        line_t tmp = src[i];
        for (int j = 0; j < BOARD_SIZE; j++) {
            dest[i][j] = tmp & 3;
            tmp >>= 2;
        }
    }
}

/// @brief read and modify compressed board
#define get_xy(arr, x, y)      (int)((arr[x] >> ((y) * 2)) & 3)
#define set_xy(arr, x, y, v)   arr[x] += ((v) - get_xy(arr, x, y)) * (1 << ((y) * 2))
#define add_xy(arr, x, y, v)   arr[x] += (((v) << ((y) * 2)))
#define minus_xy(arr, x, y, v) arr[x] -= (((v) << ((y) * 2)))
#define get(arr, p)            get_xy(arr, p.x, p.y)
#define set(arr, p, v)         set_xy(arr, p.x, p.y, v)
#define add(arr, p, v)         add_xy(arr, p.x, p.y, v)
#define minus(arr, p, v)       minus_xy(arr, p.x, p.y, v)

#ifdef BOTZONE
#    define MAX_TREE_SIZE 1001000
#    define NODE_LIMIT    1000000
#else
#    define MAX_TREE_SIZE 20001000
#    define NODE_LIMIT    20000000
#endif

// node_t* memory_pool;

void print_cprboard(const cprboard_t board, point_t emph_pos)
{
    board_t b;
    decode(board, b);
    emph_print(b, emph_pos);
}

/// @brief print data of status {st}
void print_status(const status_t st)
{
    log("print status:");
    log("board:");
    print_cprboard(st.board, st.pos);
    log("visited:");
    print_cprboard(st.visited, st.pos);
    log("id: %d, score: %d", st.id, st.score);
    log("result: %d, count: %d", st.result, st.count);
    log("begin: (%d, %d), end: (%d, %d), capacity: %d", st.begin.x, st.begin.y, st.end.x, st.end.y,
        st.capacity);
    log("piece_cnt: %d, visited_cnt: %d", st.piece_cnt, st.visited_cnt);
    log("pos: (%d, %d)", st.pos.x, st.pos.y);
    for (int i = 0; i < 2; i++) {
        log("win pos #%d: (%d, %d)", i, st.win_pos[i].x, st.win_pos[i].y);
    }
    log("done.");
}

/// @brief get pattern by compressed board {bd} and position {pos}
/// @return pattern type
static pattern4_t get_pattern4(const cprboard_t bd, point_t pos)
{
    int id;
    if (!inboard(pos) || !(id = get(bd, pos))) return PAT4_OTHERS;
    const int8_t arrows[4][2] = {{1, 1}, {1, -1}, {1, 0}, {0, 1}};
    int idx[4];
    int piece, val;
    for (int8_t i = 0, dx, dy; i < 4; i++) {
        dx = arrows[i][0], dy = arrows[i][1];
        val = 0;
        for (int8_t j = -WIN_LENGTH + 1; j < WIN_LENGTH; j++) {
            point_t np = (point_t){pos.x + dx * j, pos.y + dy * j};
            if (!inboard(np))
                val = val * PIECE_SIZE + OPPO_PIECE;
            else if (!(piece = get(bd, np)))
                val = val * PIECE_SIZE + EMPTY_PIECE;
            else
                val = val * PIECE_SIZE + ((piece == id) ? SELF_PIECE : OPPO_PIECE);
        }
        idx[i] = to_pattern(val);
    }
    pattern4_t pat4 = to_pattern4(idx[0], idx[1], idx[2], idx[3]);
    if (!param.check_forbid) {
        if (pat4 > PAT4_WIN) {
            if (pat4 == PAT4_TL) {
                pat4 = PAT4_WIN;
            } else {
                pat4 = PAT4_OTHERS;
            }
        }
    }
    return pat4;
}

/// @brief get pattern after putting a piece of player{id} at {pos}
/// @return pattern type
static pattern4_t get_virtual_pattern4(cprboard_t bd, point_t pos, int id)
{
    assert(inboard(pos) && !get(bd, pos));
    add(bd, pos, id);
    pattern4_t pat4 = get_pattern4(bd, pos);
    minus(bd, pos, id);
    return pat4;
}

/// @brief check if {pos} is forbidden for player{id}
/// @return 0 if not forbidden, pattern4 type otherwise.
static int compressive_forbidden(cprboard_t bd, point_t pos, int id)
{
    assert(inboard(pos));
    if (!param.check_forbid) return 0;
    if (id != first_id) return 0;
    pattern4_t pat4 = get_virtual_pattern4(bd, pos, id);
    if (pat4 <= PAT4_WIN) return 0;
    return pat4;
}

/// @brief get positions which are winning pos after last move
/// @param pos position of last piece
static void get_win_pos(status_t* st, point_t pos)
{
    line_t* board = st->board;
    int id = st->id, oppo = 3 - st->id;
    if (!id) {
        return;
    }
    static const int8_t arrows[4][2] = {{0, 1}, {1, 0}, {1, 1}, {-1, 1}};
    point_t tmp = pos;
    memset(st->win_pos, -1, sizeof(st->win_pos));
    // log("start.");
    int8_t dx, dy;
    int mid = WIN_LENGTH - 1;  // . . . . X . . . .
    for (int i = 0, val, pattern, piece, cnt = 0; i < 4 && cnt < WINPOS_SIZE; i++) {
        dx = arrows[i][0], dy = arrows[i][1];
        val = 0;
        for (int j = -mid; j <= mid; j++) {
            tmp = (point_t){pos.x + dx * j, pos.y + dy * j};
            piece = inboard(tmp) ? get(board, tmp) : oppo;
            val = val * PIECE_SIZE +
                  (piece == id ? SELF_PIECE : (piece == oppo ? OPPO_PIECE : EMPTY_PIECE));
        }
        pattern = to_pattern(val);
        if (pattern == PAT_D4 || pattern == PAT_A4) {
            int col[2];
            get_upgrade_columns(val, col, 2);
            for (int j = 0; j < 2 && cnt < WINPOS_SIZE; j++) {
                if (col[j] != -1) {
                    tmp = (point_t){pos.x + dx * (col[j] - mid), pos.y + dy * (col[j] - mid)};
                    assert(inboard(tmp) && !get(board, tmp));
                    if (!compressive_forbidden(st->board, tmp, id)) {
                        st->win_pos[cnt++] = tmp;
                    }
                }
            }
        }
    }
}

/// @brief create status from given info
/// @param board compressed board
/// @param begin left-top corner of the area of board
/// @param end right-bottom corner of the area of board
/// @param cnt piece count
static status_t create_status(cprboard_t board, point_t begin, point_t end, point_t pos, int cnt,
                              int next_id)
{
    status_t st = {0};
    memcpy(st.board, board, sizeof(st.board));
    memcpy(st.visited, board, sizeof(st.board));
    st.piece_cnt = cnt;
    st.visited_cnt = st.piece_cnt;
    st.begin = begin;
    st.end = end;
    st.id = next_id;
    st.capacity = ((int)end.x - begin.x) * ((int)end.y - begin.y);

    int prev_id = 3 - st.id;
    int pattern = get_pattern4(board, pos);
    if (prev_id != first_id && pattern == PAT4_TL) pattern = PAT4_WIN;
    if (pattern == PAT4_WIN) {
        st.score = (prev_id == 1) ? 1 : -1;
    }

    st.pos = pos;
    get_win_pos(&st, pos);
    return st;
}

/// @brief update {status} with new piece at {pos}
/// @param pattern pre-calculated pattern
/// @return updated status
static status_t update_status(status_t status, point_t pos, int pattern)
{
    add(status.board, pos, status.id);
    memcpy(status.visited, status.board, sizeof(status.board));
    status.pos = pos;
    if (param.dynamic_area) {
        chkmin(status.begin.x, max(0, pos.x - param.wrap_rad));
        chkmin(status.begin.y, max(0, pos.y - param.wrap_rad));
        chkmax(status.end.x, min(BOARD_SIZE - 1, pos.x + param.wrap_rad));
        chkmax(status.end.y, min(BOARD_SIZE, pos.y + param.wrap_rad));
    }
    status.piece_cnt++;
    status.visited_cnt = status.piece_cnt;
    if (status.id != first_id && pattern == PAT4_TL) pattern = PAT4_WIN;
    if (pattern == PAT4_WIN) {
        status.score = (status.id == 1) ? 1 : -1;
        // log("win situation: ");
        // print_cprboard(status.board, pos);
        // prompt_pause();
    }
    get_win_pos(&status, pos);
    status.id = 3 - status.id;
    return status;
}

/// @brief create a node containing {status}
/// @param status status of the node
static node_t* create_node(status_t status)
{
    if (tot >= NODE_LIMIT) return NULL;
    node_t* node;
    // node = (node_t*)malloc(sizeof(node_t));
    node = node_buffer + tot++;
    memset(node, 0, sizeof(node_t));
    memcpy(&node->status, &status, sizeof(status_t));
    return node;
}

/// @brief set {child} to be one of the children of {node}
/// @return the count of {node}'s children
static int append_child(node_t* node, node_t* child)
{
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
static double ucb_eval(node_t* parent, node_t* node, int flag)
{
    int win_cnt = node->status.count + flag * node->status.result;
    double f1 = (double)win_cnt / node->status.count;
    double f2 = sqrt(log(parent->status.count) / node->status.count);
    return f1 + param.C * f2;
}
#define log log_l

/// @brief print top {count} candidates of a node
static void print_candidate(node_t* parent, int count)
{
    if (parent->child_edge == NULL) return;
    edge_t* cur = parent->child_edge;
    node_t** cand = (node_t**)malloc(count * sizeof(node_t*));
    cand[0] = cur->to;
    int cnt = 0;
    while (cur != NULL) {
        if (cur->to->status.count > cand[0]->status.count) {
            for (int i = count - 1; i > 0; i--) {
                cand[i] = cand[i - 1];
            }
            cand[0] = cur->to;
            if (cnt < count) cnt++;
        }
        cur = cur->next;
    }
#define percentage(a, b) ((double)(a) / (double)(b) * 100)
#define get_rate(st, f)  (percentage(((double)st.count + (f) * st.result) / 2, st.count))
#define print_stat(i, st)                                                                          \
    log("(%hhd, %hhd) -> win: %.2lf%%, count: %.1lf%% (%d), eval: %.3lf", st.pos.x, st.pos.y,      \
        get_rate(st, parent->status.id == 1 ? 1 : -1), percentage(st.count, parent->status.count), \
        st.count, ucb_eval(parent, cand[i], st.id == 1 ? -1 : 1))
    for (int i = 0; i < cnt; i++) {
        if (cand[i] != NULL) {
            print_stat(i, cand[i]->status);
        }
    }
    free(cand);
#undef percentage
#undef get_rate
#undef print_stat
}

/// @brief select best child of {node} by visiting count, the node must has at least one child
static node_t* count_select(node_t* node)
{
    assert(node->child_edge);
    edge_t* cur = node->child_edge;
    node_t* child = cur->to;
    while (cur != NULL) {
        if (cur->to->status.count > child->status.count) {
            child = cur->to;
        }
        cur = cur->next;
    }
    return child;
}

/// @brief select best child of {node} by ucb value, the node must has at least one child
static node_t* ucb_select(node_t* node)
{
    assert(node->child_edge);
    edge_t* cur = node->child_edge;
    node_t* child = cur->to;
    int flag = (node->status.id == 1) ? 1 : -1;
    while (cur != NULL) {
        if (ucb_eval(node, cur->to, flag) > ucb_eval(node, child, flag)) {
            child = cur->to;
        }
        cur = cur->next;
    }
    return child;
}

/// @brief check if status is terminated
static bool terminated(status_t st)
{
    if (st.score || st.piece_cnt == st.capacity) return true;
    return false;
}

/// @brief put a piece at {pos} on board
/// @return node pointer containing new status
static node_t* put_piece(node_t* node, point_t pos)
{
    add(node->status.visited, pos, 1);
    node->status.visited_cnt++;

    add(node->status.board, pos, node->status.id);
    int new_pattern = get_pattern4(node->status.board, pos);
    minus(node->status.board, pos, node->status.id);
    if (node->status.id == first_id && new_pattern > PAT4_WIN) return NULL;

    node_t* child = create_node(update_status(node->status, pos, new_pattern));
    if (child != NULL) append_child(node, child);
    return child;
}

/// @brief find a child of {node} which is at {pos}, if no such child, then create one
/// @return child at pos
static node_t* find_child(node_t* node, point_t pos)
{
    // print_cprboard(parent->status.board, pos);
    // log("id: %d", parent->status.id);
    // prompt_pause();
    for (edge_t* edge = node->child_edge; edge != NULL; edge = edge->next) {
        if (edge->to->status.pos.x == pos.x && edge->to->status.pos.y == pos.y) {
            return edge->to;
        }
    }
    return put_piece(node, pos);
}

/// @brief traverse the tree to find a leaf node
static node_t* traverse(node_t* node)
{
    if (node == NULL || terminated(node->status)) {
        return node;
    }
    status_t status = node->status;
    int id = status.id;
    node_t* parent = node->parent;
    point_t danger_pos, win_pos;
    for (int i = 0; i < 2; i++) {
        win_pos = parent->status.win_pos[i];
        if (inboard(win_pos) && get(status.board, win_pos) == 0) {
            assert(!compressive_forbidden(status.board, win_pos, id));
            return traverse(find_child(node, win_pos));
        }
    }
    for (int i = 0; i < 2; i++) {
        danger_pos = status.win_pos[i];
        if (inboard(danger_pos)) {
            if (!compressive_forbidden(status.board, danger_pos, id)) {
                return traverse(find_child(node, danger_pos));
            }
        }
    }
    // log("%d", parent->status.id);
    int res = status.capacity - status.visited_cnt;
    if (res) {
        int index = (rand() % res) + 1;
        int8_t i = status.begin.x, j = status.begin.y;
        for (int t = 0, cnt = 0; t < status.capacity; t++, j++) {
            if (j >= status.end.y) i++, j = status.begin.y;
            point_t pos = (point_t){i, j};
            if (!get(status.visited, pos)) cnt++;
            if (cnt == index) {
                // log("res = %d, pos = %d, choose %d, %d", res, pos, i, j);
                return traverse(put_piece(node, pos));
            }
        }
        return node;
    } else {
        if (node->child_cnt)
            return traverse(ucb_select(node));
        else {
            node->status.score = id == 1 ? -1 : 1;  // no available space
            // print_status(node->status), log("no available space"), prompt_pause();
            return node;
        }
    }
}

/// @brief backpropagate for nodes [start, end)
static void backpropagate(node_t* start, int score, node_t* end)
{
    while (start != end) {
        start->status.result += score;
        start->status.count++;
        start = start->parent;
    }
}

/// @brief get next move by mcts algorithm
point_t mcts(const game_t game, void* assets)
{
    if (game.count == 0) {
        return (point_t){BOARD_SIZE / 2, BOARD_SIZE / 2};
    }

    int start_time = record_time();

    param = *((mcts_param_t*)assets);
    if (node_buffer == NULL) node_buffer = (node_t*)malloc(MAX_TREE_SIZE * sizeof(node_t));
    if (edge_buffer == NULL) edge_buffer = (edge_t*)malloc(MAX_TREE_SIZE * sizeof(edge_t));

    srand((unsigned)time(0));

    tot = edge_tot = 0;
    first_id = game.first_id;

    point_t wrap_begin, wrap_end;
    int8_t radius = param.wrap_rad;
    if (param.dynamic_area)
        wrap_area(game.board, &wrap_begin, &wrap_end, radius);
    else {
        do {
            wrap_area(game.board, &wrap_begin, &wrap_end, radius);
        } while (((wrap_end.y - wrap_begin.y) * (wrap_end.x - wrap_begin.x) < 40) && ++radius);
    }

    cprboard_t empty_board = {0};
    node_t* root = create_node(
        create_status(empty_board, wrap_begin, wrap_end, (point_t){0, 0}, 0, game.first_id));
    for (int i = 0; i < game.count; i++) {
        root = put_piece(root, game.steps[i]);
    }
    // if (inboard(root->status.win_pos[0])) print_status(root->status), prompt_pause();

    int tim, cnt = 0;
    int wanted_count = param.min_count * (root->status.capacity + game.count * 2);
    log("simulating... (C: %.1lf~%.1lf, time: %d~%d, rad: %d, cap: %d)", param.start_c, param.end_c,
        param.min_time, game.time_limit, radius, root->status.capacity);
    // int base = 1;
    while ((tim = get_time(start_time)) < param.min_time ||
           (count_select(root)->status.count < wanted_count)) {
        if (tim > game.time_limit - 10 || tot >= NODE_LIMIT) break;
        param.C = param.start_c + (param.end_c - param.start_c) * (double)tim / game.time_limit;
        node_t* leaf = traverse(root);
        // log("traversed");
        if (leaf != NULL) {
            backpropagate(leaf, leaf->status.score, root->parent);
            cnt++;
        }
    }
    node_t* move = count_select(root);
    status_t st = move->status;
    log("simulated %d times, average %.2lf us, speed %.2lf", cnt, tim * 1000.0 / cnt,
        (double)cnt / tim);
    log("consumption: %d nodes, %d ms", tot, get_time(start_time));
    // print_candidate(root, 5);
    int level, f = game.cur_id == 1 ? 1 : -1;
    double rate = ((double)(st.count + (f)*st.result) / 2 / st.count * 100);
    if (rate < 85 && rate > 15)
        level = PROMPT_LOG;
    else
        level = PROMPT_WARN;
    log_add(level, "(%d, %d) -> win: %.2lf%%, count: %.2lf%% (%d).", st.pos.x, st.pos.y, rate,
            (double)st.count / root->status.count * 100, st.count);
    return st.pos;
}

point_t mcts_nn(const game_t game, void* assets)
{
    (void)game, (void)assets;
    log_e("not implemented!"), prompt_pause();
    return (point_t){GAMECTRL_GIVEUP, GAMECTRL_GIVEUP};
}

#ifdef TEST
#    include "tests/mcts.txt"
#endif