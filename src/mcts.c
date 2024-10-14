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
    point_t win_pos[2];
    point_t begin; // wrap left-top
    point_t end; // wrap right-bottom
    int capacity;
    int pattern;
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
            if (src[i][j] > 2) dest[i] -= 2;
        }
    }
}

/// @brief decode from compressed board to raw board
static void decode(const cprboard_t src, board_t dest)
{
    for (int i = 0; i < BOARD_SIZE; i++) {
        cpr_t tmp = src[i];
        for (int j = 0; j < BOARD_SIZE; j++) {
            dest[i][j] = tmp & 3;
            tmp >>= 2;
        }
    }
}

/// @brief read and modify compressed board
#define get_xy(arr, x, y)      (int)((arr[x] >> ((y) * 2)) & 3)
#define set_xy(arr, x, y, v)   arr[x] += ((v) - get_xy(arr, x, y)) * (1 << ((y) * 2))
#define add_xy(arr, x, y, v)   arr[x] += ((v) * (1 << ((y) * 2)))
#define minus_xy(arr, x, y, v) arr[x] -= ((v) * (1 << ((y) * 2)))
#define get(arr, p)            get_xy(arr, p.x, p.y)
#define set(arr, p, v)         set_xy(arr, p.x, p.y, v)
#define add(arr, p, v)         add_xy(arr, p.x, p.y, v)
#define minus(arr, p, v)       minus_xy(arr, p.x, p.y, v)

#define MAX_TREE_SIZE 20001000
#define NODE_LIMIT    20000000

// node_t* memory_pool;

void print_cprboard(const cprboard_t board, point_t emph_pos)
{
    board_t b;
    decode(board, b);
    emph_print(b, emph_pos);
}

/// @brief print board for certain status
static void print_status(const status_t st)
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

/// @brief check if has winner on pos, same as board.c:check()
static int compressive_check_backup(const cprboard_t bd, point_t pos)
{
    int id = get(bd, pos);
    if (!id) return 0;
    static const int8_t arrows[4][2] = {{0, 1}, {1, 0}, {1, 1}, {-1, 1}};
    int8_t dx, dy;
    for (int i = 0, cnt; i < 4; i++) {
        dx = arrows[i][0], dy = arrows[i][1];
        point_t np = {pos.x, pos.y};
        for (cnt = 0; inboard(np); np.x += dx, np.y += dy) {
            if (get(bd, np) == id)
                cnt++;
            else
                break;
        }
        np = (point_t){pos.x - dx, pos.y - dy};
        for (; inboard(np); np.x -= dx, np.y -= dy) {
            if (get(bd, np) == id)
                cnt++;
            else
                break;
        }
        if (cnt >= WIN_LENGTH) return id == 1 ? 1 : -1;
    }
    return 0;
}

static int get_pattern4(const cprboard_t bd, point_t pos)
{
    int id;
    if (!inboard(pos) || !(id = get(bd, pos))) return PAT4_OTHERS;
    static int8_t arrow[4][2] = {{1, 1}, {1, -1}, {1, 0}, {0, 1}};
    static int idx[4];
    int piece, val;
    for (int8_t i = 0, a, b; i < 4; i++) {
        a = arrow[i][0], b = arrow[i][1];
        val = 0;
        for (int8_t j = -WIN_LENGTH + 1; j < WIN_LENGTH; j++) {
            point_t np = (point_t){pos.x + a * j, pos.y + b * j};
            if (!inboard(np))
                val = val * 3 + OPPO_POS;
            else if (!(piece = get(bd, np)))
                val = val * 3 + EMPTY_POS;
            else
                val = val * 3 + ((piece == id) ? SELF_POS : OPPO_POS);
        }
        // print_segment(seg);
        idx[i] = to_pattern(val);
    }
    // printf("%s | %s | %s | %s\n", pattern_typename[idx[0]], pattern_typename[idx[1]],
    //        pattern_typename[idx[2]], pattern_typename[idx[3]]);
    int pat4 = to_pattern4(idx[0], idx[1], idx[2], idx[3]);
    if (!param.check_ban) {
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

static int get_virtual_pattern4(cprboard_t bd, point_t pos, int id)
{
    assert(inboard(pos) && !get(bd, pos));
    add(bd, pos, id);
    int pat4 = get_pattern4(bd, pos);
    minus(bd, pos, id);
    return pat4;
}

static int compressive_banned(cprboard_t bd, point_t pos, int id)
{
    assert(inboard(pos));
    if (!param.check_ban) return 0;
    if (id != first_id) return 0;
    int pat4 = get_virtual_pattern4(bd, pos, id);
    if (pat4 <= PAT4_WIN) return 0;
    return pat4;
}

static int compressive_check(const cprboard_t bd, point_t pos)
{
    int id = get(bd, pos);
    if (!id) return 0;
    int pat4 = get_pattern4(bd, pos), result = 0;
    if (pat4 == PAT4_WIN || (id != first_id && pat4 == PAT4_TL)) {
        result = (id == 1) ? 1 : -1;
    }
#ifdef DEBUG
    int result2 = compressive_check_backup(bd, pos);
    if (result != result2) {
        print_cprboard(bd, pos);
        log("pos: (%d, %d)", pos.x, pos.y);
        log("confilct %d <-> %d", result, result2);
        prompt_pause();
    }
#endif
    return result;
}

/// @brief get 2 positions which are winning pos for the opponent
static void get_win_pos(status_t* st, point_t pos)
{
    cpr_t* board = st->board;
    int id = st->id, oppo = 3 - st->id;
    // st->danger_pos[0] = st->danger_pos[1] = (point_t){-1, -1};
    if (!id) {
        return;
    }
    static const int8_t arrows[4][2] = {{0, 1}, {1, 0}, {1, 1}, {-1, 1}};
    point_t tmp = pos;
    memset(st->win_pos, -1, sizeof(st->win_pos));
    // log("start.");
    int8_t dx, dy;
    int mid = WIN_LENGTH - 1;  // . . . . X . . . .
    for (int i = 0, val, pattern, piece; i < 4; i++) {
        dx = arrows[i][0], dy = arrows[i][1];
        val = 0;
        for (int j = -mid; j <= mid; j++) {
            tmp = (point_t){pos.x + dx * j, pos.y + dy * j};
            piece = inboard(tmp) ? get(board, tmp) : oppo;
            val = val * 3 + (piece == id ? SELF_POS : (piece == oppo ? OPPO_POS : EMPTY_POS));
        }
        pattern = to_pattern(val);
        if (pattern == PAT_D4 || pattern == PAT_A4) {
            int col[2];
            get_critical_column(val, col, 2);
            for (int j = 0; j < 2; j++) {
                if (col[j] != -1) {
                    tmp = (point_t){pos.x + dx * (col[j] - mid), pos.y + dy * (col[j] - mid)};
                    assert(inboard(tmp) && !get(board, tmp));
                    if (!compressive_banned(st->board, tmp, id)) {
                        for (int k = 0; k < 2; k++) {
                            if (st->win_pos[k].x == -1) {
                                st->win_pos[k] = tmp;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    // if (inboard(pos0)) {
    //     log("origin pos: (%d, %d)", st->pos.x, st->pos.y);
    //     for (int i = 0; i < 2; i++) {
    //         if (inboard(st->win_pos[i])) {
    //             log("get win pos for %d: (%d, %d)", id, st->danger_pos[i].x,
    //             st->danger_pos[i].y); print_cprboard(st->board, st->win_pos[i]); prompt_pause();
    //         }
    //     }
    // }
}

/// @brief create status from given info
/// @param board compressed board
/// @param begin left-top position of the area of board
/// @param end right-bottom position of the area of board
/// @param pos position of the piece
/// @param cnt piece count
/// @param next_id next id
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
    st.pattern = get_pattern4(board, pos);
    if (prev_id != first_id && st.pattern == PAT4_TL) st.pattern = PAT4_WIN;
    if (st.pattern == PAT4_WIN) {
        st.score = (prev_id == 1) ? 1 : -1;
    }

    st.pos = pos;
    get_win_pos(&st, pos);
    return st;
}

/// @brief update status with new piece
/// @param status status to update
/// @param pos position of the new piece
/// @param calculated_pattern pattern of the new piece
/// @return updated status
static status_t update_status(status_t status, point_t pos, int calculated_pattern)
{
    add(status.board, pos, status.id);
    memcpy(status.visited, status.board, sizeof(status.board));
    status.pos = pos;
    status.piece_cnt++;
    status.visited_cnt = status.piece_cnt;
    // status.pattern = get_pattern4(status.board, pos);
    status.pattern = calculated_pattern;
    if (status.id != first_id && status.pattern == PAT4_TL) status.pattern = PAT4_WIN;
    if (status.pattern == PAT4_WIN) {
        status.score = (status.id == 1) ? 1 : -1;
        // log("win situation: ");
        // print_cprboard(status.board, pos);
        // prompt_pause();
    }
    get_win_pos(&status, pos);
    status.id = 3 - status.id;
    return status;
}

/// @brief create node from given status
/// @param status status of the node
static node_t* create_node(status_t status)
{
    if (tot >= NODE_LIMIT) return NULL;
    node_t* node;
    // if ((node = hashmap_get(hashmap, status.hash)) != NULL) {
    //     // exit(0);
    //     // board_t bd;
    //     // decode(status.board, bd);
    //     // log("status"), print(bd);
    //     // decode(node->status.board, bd);
    //     // log("node->status"), print(bd);
    //     // getchar();
    //     reused_tot++;
    //     node->status.pos = status.pos;
    //     // get_danger_pos(&node->status, status.pos);
    //     return node;
    // }

    // node = (node_t*)malloc(sizeof(node_t));
    node = node_buffer + tot++;
    memset(node, 0, sizeof(node_t));
    memcpy(&node->status, &status, sizeof(status_t));
    // hashmap_insert(hashmap, status.hash, node);
    return node;
}

/// @brief append a child node to parent node
/// @param parent parent node
/// @param node child node
/// @return the count of children of parent node
static int append_child(node_t* parent, node_t* node)
{
    edge_t* child_edge = edge_buffer + edge_tot++;
    // edge_t* child_edge = (edge_t*)malloc(sizeof(edge_t));
    // edge_t* parent_edge = (edge_t*)malloc(sizeof(edge_t));
    memset(child_edge, 0, sizeof(edge_t));

    child_edge->next = parent->child_edge;
    child_edge->to = node;
    parent->child_edge = child_edge;
    parent->child_cnt++;

    node->parent = parent;
    return parent->child_cnt;
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

/// @brief print top <count> candidates of a node
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
#undef CAND_SIZE
#undef percentage
#undef get_rate
#undef print_stat
}

/// @brief select best child by count
static node_t* count_select(node_t* parent)
{
    if (parent->child_edge == NULL) return parent;
    edge_t* cur = parent->child_edge;
    node_t* sel = cur->to;
    while (cur != NULL) {
        if (cur->to->status.count > sel->status.count) {
            sel = cur->to;
        }
        cur = cur->next;
    }
    return sel;
}

/// @brief select best child by ucb value, return parent if no child
static node_t* ucb_select(node_t* parent)
{
    if (parent->child_cnt == 0) return parent;
    edge_t* cur = parent->child_edge;
    node_t* sel = cur->to;
    int flag = (parent->status.id == 1) ? 1 : -1;
    while (cur != NULL) {
        if (ucb_eval(parent, cur->to, flag) > ucb_eval(parent, sel, flag)) {
            sel = cur->to;
        }
        cur = cur->next;
    }
    return sel;
}

/// @brief check if status is terminated
static bool terminated(status_t st)
{
    if (st.score || st.piece_cnt == st.capacity) return true;
    return false;
}

/// @brief put a piece at pos
/// @param parent current node
/// @param pos position of the piece
/// @return pointer to the new node, NULL if requests banned position or no space
static node_t* put_piece(node_t* parent, point_t pos)
{
    add(parent->status.visited, pos, 1);
    parent->status.visited_cnt++;

    add(parent->status.board, pos, parent->status.id);
    int new_pattern = get_pattern4(parent->status.board, pos);
    minus(parent->status.board, pos, parent->status.id);
    if (parent->status.id == first_id && new_pattern > PAT4_WIN) return NULL;

    node_t* node = create_node(update_status(parent->status, pos, new_pattern));
    if (node != NULL) append_child(parent, node);
    return node;
}

/// @brief if parent has a child at pos, return the child, else create a new one
node_t* find_piece(node_t* parent, point_t pos)
{
    // print_cprboard(parent->status.board, pos);
    // log("id: %d", parent->status.id);
    // prompt_pause();
    for (edge_t* edge = parent->child_edge; edge != NULL; edge = edge->next) {
        if (edge->to->status.pos.x == pos.x && edge->to->status.pos.y == pos.y) {
            return edge->to;
        }
    }
    return put_piece(parent, pos);
}

/// @brief traverse the tree to find a leaf node (terminate status), create new
/// nodes if needed
node_t* traverse(node_t* parent)
{
    // print_status(parent->status);
    // log("%d, %d, id = %d, score = %d", parent->status.pos.x,
    // parent->status.pos.y, parent->status.id, parent->status.score);
    if (parent == NULL || terminated(parent->status)) {
        // if (parent == NULL) return NULL;
        // board_t dec;
        // decode(parent->status.board, dec);
        // print(dec);
        return parent;
    }
    status_t status = parent->status;
    int id = status.id;
    node_t* grandparent = parent->parent;
    point_t danger_pos, win_pos;
    for (int i = 0; i < 2; i++) {
        win_pos = grandparent->status.win_pos[i];
        if (inboard(win_pos) && get(status.board, win_pos) == 0) {
            assert(!compressive_banned(status.board, win_pos, id));
            // if (!compressive_banned(status.board, win_pos, id)) {
            return traverse(find_piece(parent, win_pos));
            // }
        }
    }
    for (int i = 0; i < 2; i++) {
        danger_pos = status.win_pos[i];
        if (inboard(danger_pos)) {
            // log("exit");
            // assert(!compressive_banned(status.board, danger_pos, id));
            if (!compressive_banned(status.board, danger_pos, id)) {
                return traverse(find_piece(parent, danger_pos));
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
            if (!get_xy(status.visited, i, j)) cnt++;
            if (cnt == index) {
                // log("res = %d, pos = %d, choose %d, %d", res, pos, i, j);
                return traverse(put_piece(parent, (point_t){i, j}));
            }
        }
        return parent;
    } else {
        return traverse(ucb_select(parent));
    }
}

/// @brief backpropagate the score of a node to its ancestors
void backpropagate(node_t* start, int score, node_t* end)
{
    while (start != end) {
        start->status.result += score;
        start->status.count++;
        start = start->parent;
    }
}

/// @brief mcts main function
/// @param game current game info
/// @param player_assets player's assets
point_t mcts(const game_t game, void* assets)
{
    param = *((mcts_param_t*)assets);
    if (node_buffer == NULL)
        node_buffer = (node_t*)malloc(MAX_TREE_SIZE * sizeof(node_t));
    if (edge_buffer == NULL) edge_buffer = (edge_t*)malloc(MAX_TREE_SIZE * sizeof(edge_t) * 5);

    srand((unsigned)time(0));

    int start_time = record_time();

    tot = edge_tot = 0;
    first_id = game.first_id;

    int id = game.cur_id;
    cprboard_t zip;
    memset(zip, 0, sizeof(zip));

    if (game.count == 0) {
        point_t pos = {BOARD_SIZE / 2, BOARD_SIZE / 2};
        return pos;
    }

    point_t wrap_begin, wrap_end;
    int8_t radius = param.wrap_rad;
    do {
        wrap_area(game.board, &wrap_begin, &wrap_end, radius);
    } while (((wrap_end.y - wrap_begin.y) * (wrap_end.x - wrap_begin.x) < 40) &&
             ++radius);

    node_t* root =
        create_node(create_status(zip, wrap_begin, wrap_end, (point_t){0, 0}, 0, game.first_id));
    for (int i = 0; i < game.count; i++) {
        root = put_piece(root, game.steps[i]);
    }
    // if (inboard(root->status.win_pos[0])) print_status(root->status), prompt_pause();

    int tim, cnt = 0;
    int wanted_count = param.min_count * (root->status.capacity + game.count * 2);
    log("searching... (C: %.1lf~%.1lf, time: %d~%d, target: %d, rad: %d, cap: %d)", param.start_c,
        param.end_c, param.min_time, game.time_limit, wanted_count, radius, root->status.capacity);
    // int base = 1;
    while ((tim = get_time(start_time)) < param.min_time ||
           (count_select(root)->status.count < wanted_count)) {
        if (tim > game.time_limit - 30 || tot >= NODE_LIMIT) break;
        param.C = param.start_c + (param.end_c - param.start_c) * (double)tim / game.time_limit;
        node_t* leaf = traverse(root);
        // log("traversed");
        if (leaf != NULL) {
            backpropagate(leaf, leaf->status.score, root->parent);
            cnt++;
        }
        // if (cnt % base == 0) {
        //     log("tried %d times.", cnt);
        //     base *= 2;
        // }
    }
    log("all count: %d, speed: %.2lf", cnt, (double)cnt / get_time(start_time));
    log("consumption: %d nodes, %d edges, %d ms", tot, edge_tot, get_time(start_time));
    print_candidate(root, 5);
    node_t* move = count_select(root);
    log("clear cache");
    status_t st = move->status;
    int f = id == 1 ? 1 : -1;
    double rate = ((double)(st.count + (f)*st.result) / 2 / st.count * 100);
    if (rate < 80 && rate > 20)
        log("(%d, %d) => win: %.2lf%%, count: %.2lf%% (%d).", st.pos.x, st.pos.y, rate,
            (double)st.count / root->status.count * 100, st.count);
    else
        log_w("(%d, %d) => win: %.2lf%%, count: %.2lf%% (%d).", st.pos.x, st.pos.y, rate,
              (double)st.count / root->status.count * 100, st.count);
    // assert(!get(root->status.board, st.pos));
    return st.pos;
}
