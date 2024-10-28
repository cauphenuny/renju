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

typedef line_t comp_board_t[BOARD_SIZE];  // compressed board

#define WINPOS_SIZE 2

typedef struct {
    comp_board_t board, visited;
    int piece_cnt, visited_cnt;
    int result, count;
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
static int first_id;
static int start_time;
static mcts_param_t param;

/// @brief encode from raw board to compressed board
static void encode(const board_t src, comp_board_t dest)
{
    for (int i = 0; i < BOARD_SIZE; i++) {
        dest[i] = 0;
        for (int j = BOARD_SIZE - 1; j >= 0; j--) {
            dest[i] = dest[i] * 4 + src[i][j];
        }
    }
}

/// @brief decode from compressed board to raw board
static void decode(const comp_board_t src, board_t dest)
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
#    define MAX_TREE_SIZE 30001000
#    define NODE_LIMIT    30000000
#endif

// node_t* memory_pool;

static void print_compressed_board(const comp_board_t board, point_t emph_pos)
{
    board_t b;
    decode(board, b);
    emphasis_print(b, emph_pos);
}

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

static void visualize_children(const node_t* node)
{
    if (!node->state.count) return;
    fboard_t prob = {0};
    int sum = 0;
    for (const edge_t* e = node->child_edge; e; e = e->next) {
        const node_t* child = e->to;
        prob[child->state.pos.x][child->state.pos.y] =
            (double)child->state.count / node->state.count;
        sum += child->state.count;
    }
    if (param.prob_matrix != NULL) {
        memcpy(param.prob_matrix, prob, sizeof(prob));
    }
    board_t board;
    decode(node->state.board, board);
    probability_print(board, prob);
    assert(sum == node->state.count);
}

/// @brief get pattern by compressed board {bd} and position {pos}
/// @return pattern type
static pattern4_t pattern4_type_comp(const comp_board_t board, point_t pos)
{
    int id;
    if (!in_board(pos) || !((id = get(board, pos)))) return PAT4_OTHERS;
    const int8_t arrows[4][2] = {{1, 1}, {1, -1}, {1, 0}, {0, 1}};
    int idx[4];
    int piece;
    for (int8_t i = 0, dx, dy; i < 4; i++) {
        dx = arrows[i][0], dy = arrows[i][1];
        int val = 0;
        for (int8_t j = -WIN_LENGTH + 1; j < WIN_LENGTH; j++) {
            const point_t np = (point_t){pos.x + dx * j, pos.y + dy * j};
            if (!in_board(np))
                val = val * PIECE_SIZE + OPPO_PIECE;
            else if (!((piece = get(board, np))))
                val = val * PIECE_SIZE + EMPTY_PIECE;
            else
                val = val * PIECE_SIZE + ((piece == id) ? SELF_PIECE : OPPO_PIECE);
        }
        idx[i] = to_pattern(val);
    }
    pattern4_t pat4 = to_pattern4(idx[0], idx[1], idx[2], idx[3]);
    if (!param.check_forbid || id != first_id) {
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
static pattern4_t virtual_pat4type_comp(comp_board_t board, point_t pos, int id)
{
    assert(in_board(pos) && !get(board, pos));
    add(board, pos, id);
    const pattern4_t pat4 = pattern4_type_comp(board, pos);
    minus(board, pos, id);
    return pat4;
}

/// @brief check if {pos} is forbidden for player{id}
/// @return 0 if not forbidden, pattern4 type otherwise.
static int is_forbidden_comp(comp_board_t bd, point_t pos, int id)
{
    assert(in_board(pos));
    if (!param.check_forbid) return 0;
    if (id != first_id) return 0;
    const pattern4_t pat4 = virtual_pat4type_comp(bd, pos, id);
    if (pat4 <= PAT4_WIN) return 0;
    return pat4;
}

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
    const int8_t arrows[4][2] = {{0, 1}, {1, 0}, {1, 1}, {-1, 1}};
    point_t tmp;
    // print_state(*st), log("start");
    int8_t dx, dy;
    const int mid = WIN_LENGTH - 1;  // . . . . X . . . .
    for (int i = 0, val, pattern, piece, cnt = 0; i < 4 && cnt < WINPOS_SIZE; i++) {
        dx = arrows[i][0], dy = arrows[i][1];
        val = 0;
        for (int j = -mid; j <= mid; j++) {
            tmp = (point_t){pos.x + dx * j, pos.y + dy * j};
            piece = in_board(tmp) ? get(board, tmp) : oppo;
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
                    assert(in_board(tmp) && !get(board, tmp));
                    if (!is_forbidden_comp(st->board, tmp, id)) {
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
static state_t empty_state(point_t begin, point_t end, int next_id)
{
    state_t st = {0};
    st.begin = begin;
    st.end = end;
    st.id = next_id;
    st.capacity = ((int)end.x - begin.x) * ((int)end.y - begin.y);
    st.pos = (point_t){-1, -1};
    get_win_pos(&st);
    return st;
}

/// @brief update {state} with new piece at {pos}
/// @param new_pat4 pre-calculated pattern
/// @return updated state
static state_t update_state(state_t state, point_t pos, pattern4_t new_pat4)
{
    add(state.board, pos, state.id);
    memcpy(state.visited, state.board, sizeof(state.board));
    state.pos = pos;
    if (in_area(pos, state.begin, state.end)) {
        state.piece_cnt++;
    }
    if (param.dynamic_area) {
        chkmin(state.begin.x, max(0, pos.x - param.wrap_rad));
        chkmin(state.begin.y, max(0, pos.y - param.wrap_rad));
        chkmax(state.end.x, min(BOARD_SIZE - 1, pos.x + param.wrap_rad));
        chkmax(state.end.y, min(BOARD_SIZE, pos.y + param.wrap_rad));
    }
    state.count = 0, state.result = 0;
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
    return state;
}

/// @brief create a node containing {state}
/// @param state state of the node
static node_t* create_node(state_t state)
{
    if (tot >= NODE_LIMIT) return NULL;
    node_t* node = node_buffer + tot++;
    // node = (node_t*)malloc(sizeof(node_t));
    memset(node, 0, sizeof(node_t));
    memcpy(&node->state, &state, sizeof(state_t));
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
static double ucb_eval(const node_t* node)
{
    const int flag = (node->parent->state.id == 1) ? 1 : -1;
    const int win_cnt = node->state.count + flag * node->state.result;
    const double f1 = (double)win_cnt / node->state.count;
    const double f2 = sqrt(log(node->parent->state.count) / node->state.count);
    return f1 + param.C * f2;
}

static double entropy(const node_t* node)
{
    double sum = 0;
    for (const edge_t* e = node->child_edge; e; e = e->next) {
        const double prob = (double)e->to->state.count / node->state.count;
        sum += prob * log(1 / prob);
    }
    return sum;
}
#define log log_l

/// @brief select best child of {node} by visiting count, the node must have at least one child
static node_t* max_count(node_t* node)
{
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
static node_t* min_count(node_t* node)
{
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
static node_t* ucb_select(const node_t* node)
{
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

/// @brief check if state is terminated
static bool terminated(state_t st)
{
    if (st.score || st.piece_cnt == st.capacity) return true;
    return false;
}

/// @brief put a piece at {pos} on board
/// @return node pointer containing new state
static node_t* put_piece(node_t* node, point_t pos, pattern4_t new_pat4)
{
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
static node_t* find_child(node_t* node, point_t pos)
{
    // log("id: %d", parent->state.id);
    // prompt_pause();
    for (const edge_t* edge = node->child_edge; edge != NULL; edge = edge->next) {
        if (edge->to->state.pos.x == pos.x && edge->to->state.pos.y == pos.y) {
            return edge->to;
        }
    }
    return put_piece(node, pos, virtual_pat4type_comp(node->state.board, pos, node->state.id));
}

/// @brief traverse the tree to find a leaf node
static node_t* traverse(node_t* node)
{
    if (node == NULL || terminated(node->state)) {
        return node;
    }
    state_t state = node->state;
    const int id = state.id;
    const node_t* parent = node->parent;
    for (int i = 0; i < 2; i++) {
        const point_t win_pos = parent->state.win_pos[i];
        if (in_board(win_pos) && get(state.board, win_pos) == 0) {
            assert(!is_forbidden_comp(state.board, win_pos, id));
            return traverse(find_child(node, win_pos));
        }
    }
    for (int i = 0; i < 2; i++) {
        const point_t danger_pos = state.win_pos[i];
        if (in_board(danger_pos)) {
            if (!is_forbidden_comp(state.board, danger_pos, id)) {
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
                pat4 = virtual_pat4type_comp(state.board, pos, id);
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
        return traverse(ucb_select(node));
    }
    node->state.score = id == 1 ? -1 : 1;  // no available space
    return node;
}

static void simulate(node_t* start_node)
{
    node_t* leaf = traverse(start_node);
    if (leaf == NULL) return;
    const int score = leaf->state.score;
    while (leaf != NULL) {
        leaf->state.result += score;
        leaf->state.count++;
        assert(leaf->state.result <= leaf->state.count && -leaf->state.result <= leaf->state.count);
        leaf = leaf->parent;
    }
}

/// @brief get next move by mcts algorithm
point_t mcts(const game_t game, void* assets)
{
    if (game.count == 0) {
        return (point_t){BOARD_SIZE / 2, BOARD_SIZE / 2};
    }

    start_time = record_time();

    param = *((mcts_param_t*)assets);
    if (node_buffer == NULL) node_buffer = (node_t*)malloc(MAX_TREE_SIZE * sizeof(node_t));
    if (edge_buffer == NULL) edge_buffer = (edge_t*)malloc(MAX_TREE_SIZE * sizeof(edge_t));

    srand(time(0));

    tot = edge_tot = 0;
    first_id = game.first_id;

    point_t wrap_begin, wrap_end;
    int radius = param.wrap_rad;
    if (param.dynamic_area)
        wrap_area(game.board, &wrap_begin, &wrap_end, radius);
    else {
        do {
            wrap_area(game.board, &wrap_begin, &wrap_end, radius);
        } while (((wrap_end.y - wrap_begin.y) * (wrap_end.x - wrap_begin.x) < 80) && ++radius);
    }

    node_t* root = create_node(empty_state(wrap_begin, wrap_end, game.first_id));
    for (int i = 0; i < game.count; i++) {
        root = put_piece(root, game.steps[i], PAT4_OTHERS);
    }
    for (int i = 0; i < WINPOS_SIZE; i++) {
        const point_t pos = root->parent->state.win_pos[i];
        if (in_board(pos)) {
            log("(%d, %d)", pos.x, pos.y);
        }
    }

    int tim, cnt = 0;
    const int target_count = param.min_count * (root->state.capacity + game.count * 2);

    log("simulating (C: %.1lf~%.1lf, time: %d~%d, rad: %d)", param.start_c, param.end_c, param.min_time, game.time_limit, radius);
    // int base = 1;
    while ((tim = get_time(start_time)) < param.min_time ||
           max_count(root)->state.count < target_count) {
        if (tim > game.time_limit - 10 || tot >= NODE_LIMIT) break;
        param.C = param.start_c + (param.end_c - param.start_c) * (double)tim / game.time_limit;
        simulate(root), cnt++;
    }
    log("simulated %d times, average %.2lf us, speed %.2lf", cnt, tim * 1000.0 / cnt,
        (double)cnt / tim);
    log("consumption: %d nodes, %d ms", tot, get_time(start_time));
    log("entropy: %.3lf, min count: %d, max count: %d", entropy(root), min_count(root)->state.count, max_count(root)->state.count);
    log("visualize:");
    visualize_children(root);

    const node_t* move = max_count(root);
    const state_t st = move->state;
    int level, f = game.cur_id == 1 ? 1 : -1;
    const double rate = ((double)(st.count + f * st.result) / 2 / st.count * 100);
    if (rate < 85 && rate > 15)
        level = PROMPT_LOG;
    else
        level = PROMPT_WARN;
    log_add(level, "(%d, %d) -> prob: %.2lf%% (%d), win: %.2lf%%.", st.pos.x, st.pos.y,
            (double)st.count / root->state.count * 100, st.count, rate);
#ifdef TEST
    point_t manual(const game_t, void*);
    if (level == PROMPT_WARN) {
        print_state(root->state);
        while (1) {
            const point_t pos = manual(game, assets);
            if (in_board(pos)) {
                const node_t* child = find_child(root, pos);
                const double tmp = ((double)(child->state.count + (f)*child->state.result) / 2 /
                              child->state.count * 100);
                log("count: %d(%.2lf%%), win: %.2lf%%", child->state.count,
                    child->state.count * 100.0 / root->state.count, tmp);
            } else {
                break;
            }
        }
    }
#endif
    return st.pos;
}

point_t mcts_nn(const game_t game, void* assets)
{
    log_e("not implemented! call mcts without neural network instead.");
    return mcts(game, assets);
}

#ifdef TEST
#    include "tests/mcts.txt"
#endif
