// author: Cauphenuny <https://cauphenuny.github.io/>
// date: 2024/07/24
#include "mcts.h"

#include "board.h"
#include "util.h"
#include "zobrist.h"

#include <assert.h>
#undef log
#include <math.h>
#define log logi
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
    point_t danger_pos[2];
    zobrist_t hash;
    point_t begin; // wrap left-top
    point_t end; // wrap right-bottom
    int capacity;
} state_t;

typedef struct edge_t {
    struct node_t* to;
    struct edge_t* next;
} edge_t;

typedef struct node_t {
    state_t state;
    int child_cnt, parent_cnt;
    struct edge_t *child_edge, *parent_edge;
} node_t;

typedef struct hash_entry_t {
    zobrist_t key;
    node_t* value;
    struct hash_entry_t* next;
} hash_entry_t;

typedef struct {
    hash_entry_t** table;
} hashmap_t;

/// @brief memory buffer for node and edge, prevent from frequent malloc
node_t* node_buffer;
edge_t* edge_buffer;

int tot, edge_tot, hashmap_tot, reused_tot;
int first_id;
mcts_parm_t mcts_parm;

/// @brief encode from raw board to compressed board
void encode(const board_t src, cprboard_t dest)
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
void decode(const cprboard_t src, board_t dest)
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
#define get(arr, x, y)      (int)((arr[x] >> ((y) * 2)) & 3)
#define set(arr, x, y, v)   arr[x] += ((v) - get(arr, x, y)) * (1 << ((y) * 2))
#define add(arr, x, y, v)   arr[x] += ((v) * (1 << ((y) * 2)))
#define minus(arr, x, y, v) arr[x] -= ((v) * (1 << ((y) * 2)))

#define HASHMAP_SIZE 10000019

hashmap_t create_hashmap()
{
    hashmap_t map;
    map.table = (hash_entry_t**)calloc(HASHMAP_SIZE, sizeof(hash_entry_t*));
    return map;
}

void free_hashmap(hashmap_t map)
{
    // for (int i = 0; i < HASHMAP_SIZE; i++) {
    //     hash_entry_t* entry = map.table[i];
    //     while (entry) {
    //         hash_entry_t* temp = entry;
    //         entry = entry->next;
    //         free(temp);
    //     }
    // }
    free(map.table);
}

unsigned int hash(zobrist_t key)
{
    return key % HASHMAP_SIZE;
}

/// @brief memory buffer for hash entries
hash_entry_t* hashmap_buffer;

hashmap_t hashmap;

void hashmap_insert(hashmap_t map, zobrist_t key, node_t* value)
{
    unsigned int index = hash(key);
    // hash_entry_t* new_entry =
    // (hashmap_entry_t*)malloc(sizeof(hashmap_entry_t));
    hash_entry_t* new_entry = hashmap_buffer + hashmap_tot++;
    new_entry->key = key;
    new_entry->value = value;
    new_entry->next = map.table[index];
    map.table[index] = new_entry;
}

node_t* hashmap_get(hashmap_t map, zobrist_t key)
{
    unsigned int index = hash(key);
    hash_entry_t* entry = map.table[index];
    while (entry) {
        if (entry->key == key) {
            return entry->value;
        }
        entry = entry->next;
    }
    return NULL;
}

void hashmap_remove(hashmap_t map, zobrist_t key)
{
    unsigned int index = hash(key);
    hash_entry_t* entry = map.table[index];
    hash_entry_t* prev = NULL;
    while (entry) {
        if (entry->key == key) {
            if (prev) {
                prev->next = entry->next;
            } else {
                map.table[index] = entry->next;
            }
            free(entry);
            return;
        }
        prev = entry;
        entry = entry->next;
    }
}

#define MAX_TREE_SIZE 20001000
#define NODE_LIMIT    20000000

// node_t* memory_pool;

/// @brief print board for certain state
void print_state(state_t st)
{
    board_t b;
    decode(st.board, b);
    print(b);
}

/// @brief check if has winner on pos, same as board.c:check()
int compressive_check(const cprboard_t bd, point_t pos)
{
    int id = get(bd, pos.x, pos.y);
    if (!id) return 0;
    static const int8_t arrows[4][2] = {{0, 1}, {1, 0}, {1, 1}, {-1, 1}};
    int8_t dx, dy;
    for (int i = 0, cnt; i < 4; i++) {
        dx = arrows[i][0], dy = arrows[i][1];
        point_t np = {pos.x, pos.y};
        for (cnt = 0; inboard(np); np.x += dx, np.y += dy) {
            if (get(bd, np.x, np.y) == id)
                cnt++;
            else
                break;
        }
        np = (point_t){pos.x - dx, pos.y - dy};
        for (; inboard(np); np.x -= dx, np.y -= dy) {
            if (get(bd, np.x, np.y) == id)
                cnt++;
            else
                break;
        }
        if (cnt >= WIN_LENGTH) return id == 1 ? 1 : -1;
    }
    return 0;
}

#define compressive_banned(...) POS_ACCEPT

/// @brief get 2 positions which are winning pos for the opponent
void get_danger_pos(state_t* st, point_t pos)
{
    cpr_t* bd = st->board;
    int id = get(bd, pos.x, pos.y);
    // st->danger_pos[0] = st->danger_pos[1] = (point_t){-1, -1};
    if (!id) {
        return;
    }
    static const int8_t arrows[4][2] = {{0, 1}, {1, 0}, {1, 1}, {-1, 1}};
    point_t np = pos, d0 = {-1, -1}, d1 = {-1, -1};
    // log("start.");
    int8_t dx, dy;
    for (int i = 0, cnt0 = 0, cnt1 = 0; i < 4; i++) {
        dx = arrows[i][0], dy = arrows[i][1];
        np = (point_t){pos.x, pos.y};
        for (int flag = cnt0 = cnt1 = 0; inboard(np) && flag < 2;
             np.x += dx, np.y += dy) {
            if (get(bd, np.x, np.y) == 3 - id) break;
            if (get(bd, np.x, np.y) == 0) {
                if (id == first_id &&
                    compressive_banned(bd, np, id) != POS_ACCEPT) {
                    break;
                }
                flag++;
                if (flag == 1) d1 = np;
            }
            if (flag == 0) {
                cnt0++;
                cnt1++;
            }
            if (flag == 1) {
                cnt1++;
            }
        }
        // log("cnt0: %d, cnt1: %d", cnt0, cnt1);
        np = (point_t){pos.x - dx, pos.y - dy};
        for (int flag = 0; inboard(np) && flag < 2; np.x -= dx, np.y -= dy) {
            if (get(bd, np.x, np.y) == 3 - id) break;
            if (get(bd, np.x, np.y) == 0) {
                if (id == first_id &&
                    compressive_banned(bd, np, id) != POS_ACCEPT) {
                    break;
                }
                flag++;
                if (flag == 1) d0 = np;
            }
            if (flag == 0) {
                cnt0++;
                cnt1++;
            }
            if (flag == 1) {
                cnt0++;
            }
        }
        // log("cnt0: %d, cnt1: %d", cnt0, cnt1);
        if (inboard(d0) && cnt0 >= WIN_LENGTH) {
            if (st->danger_pos[0].x == -1) {
                st->danger_pos[0] = d0;
            } else {
                if (d0.x != st->danger_pos[0].x ||
                    d0.y != st->danger_pos[0].y) {
                    st->danger_pos[1] = d0;
                }
            }
        }
        if (inboard(d1) && cnt1 >= WIN_LENGTH) {
            assert(inboard(d1));
            if (st->danger_pos[0].x == -1) {
                st->danger_pos[0] = d1;
            } else {
                if (d1.x != st->danger_pos[0].x ||
                    d1.y != st->danger_pos[0].y) {
                    st->danger_pos[1] = d1;
                }
            }
        }
    }
}

/// @brief create state from given info
/// @param board compressed board
/// @param hash zobrist hash
/// @param pos position of the piece
/// @param cnt piece count
/// @param begin left-top position of the area of board
/// @param end right-bottom position of the area of board
state_t create_state(cprboard_t board, point_t begin, point_t end,
                     zobrist_t hash, point_t pos, int cnt, int next_id)
{
    state_t st;
    memset(&st, 0, sizeof(state_t));
    memcpy(st.board, board, sizeof(st.board));
    memcpy(st.visited, board, sizeof(st.board));
    st.piece_cnt = cnt;
    st.visited_cnt = st.piece_cnt;
    st.hash = hash;
    st.begin = begin;
    st.end = end;
    st.id = next_id;
    st.capacity = ((int)end.x - begin.x) * ((int)end.y - begin.y);
    // for (int i = top; i < bottom; i++) {
    //     for (int j = left; j < right; j++) {
    //         if (st.id == first_id && !get(board, i, j) &&
    //             compressive_banned(board, (point_t){i, j}, first_id) !=
    //                 POS_ACCEPT) {
    //             add(st.visited, i, j, 1);
    //             st.visited_cnt++;
    //         }
    //     }
    // }
    // if (st.id == first_id) {
    //     st.score = compressive_check(board, pos);
    // } else {
    //     if (compressive_banned(board, pos, first_id)) {
    //         st.score = first_id == 1 ? -1 : 1;
    //     } else {
    //         st.score = compressive_check(board, pos);
    //     }
    // }
    st.score = compressive_check(board, pos);
    // if (st.score) {
    //     log("(%d, %d): %d", pos.x, pos.y, st.id);
    //     board_t bd;
    //     decode(board, bd);
    //     print(bd);
    //     getchar();
    // }
    st.pos = pos;
    st.danger_pos[0] = st.danger_pos[1] = (point_t){-1, -1};
    get_danger_pos(&st, pos);
    // log("calc danger pos: (%d, %d), all: %d\n", st.danger_pos.x,
    // st.danger_pos.y, count); if (st->danger_pos > 1) {
    //     st.score = (st.id == 1) ? -1 : 1;
    //     //print(dec_board);
    //     //logw("early terminated, score = %d", st.score);
    //     //getchar();
    // }
    return st;
}

/// @brief create node from given state
/// @param state state of the node
node_t* create_node(state_t state)
{
    if (tot >= NODE_LIMIT) return NULL;
    node_t* node;
    if ((node = hashmap_get(hashmap, state.hash)) != NULL) {
        // exit(0);
        // board_t bd;
        // decode(state.board, bd);
        // log("state"), print(bd);
        // decode(node->state.board, bd);
        // log("node->state"), print(bd);
        // getchar();
        reused_tot++;
        node->state.pos = state.pos;
        // get_danger_pos(&node->state, state.pos);
        return node;
    }

    // node = (node_t*)malloc(sizeof(node_t));
    node = node_buffer + tot++;
    memset(node, 0, sizeof(node_t));
    memcpy(&node->state, &state, sizeof(state_t));
    hashmap_insert(hashmap, state.hash, node);
    return node;
}

/// @brief append a child node to parent node
/// @param parent parent node
/// @param node child node
/// @return the count of children of parent node
int append_child(node_t* parent, node_t* node)
{
    edge_t* child_edge = edge_buffer + edge_tot++;
    edge_t* parent_edge = edge_buffer + edge_tot++;
    // edge_t* child_edge = (edge_t*)malloc(sizeof(edge_t));
    // edge_t* parent_edge = (edge_t*)malloc(sizeof(edge_t));
    memset(child_edge, 0, sizeof(edge_t));
    memset(parent_edge, 0, sizeof(edge_t));

    child_edge->next = parent->child_edge;
    child_edge->to = node;
    parent->child_edge = child_edge;
    parent->child_cnt++;

    parent_edge->next = node->parent_edge;
    parent_edge->to = parent;
    node->parent_edge = parent_edge;
    node->parent_cnt++;

    return parent->child_cnt;
}

/// @brief delete a subgraph of a node
/// @return the size of deleted subgraph
int delete_subgraph(node_t* node)
{
    int size = 1;
    node_t* child;
    for (edge_t *e = node->child_edge, *nxt; e != NULL; e = nxt) {
        nxt = e->next;
        child = e->to;
        for (edge_t *pe = child->parent_edge, *pnxt, *prev = NULL; pe != NULL;
             prev = pe, pe = pnxt) {
            pnxt = pe->next;
            if (pe->to == node) {
                if (pe != child->parent_edge) {
                    prev->next = pnxt;
                } else {
                    child->parent_edge = NULL;
                }
                free(pe);
                break;
            }
        }
        child->parent_cnt--;
        if (child->parent_cnt == 0) {
            size += delete_subgraph(child);
        }
        free(e);
    }
    hashmap_remove(hashmap, node->state.hash);
    free(node);
    return size;
}

#undef log
/// @brief get the evaluation of a node by ucb formula
double ucb_eval(node_t* parent, node_t* node, int flag)
{
    int win_cnt = node->state.count + flag * node->state.result;
    double f1 = (double)win_cnt / node->state.count;
    double f2 = sqrt(log(parent->state.count) / node->state.count);
    return f1 + mcts_parm.C * f2;
}
#define log log_l

/// @brief print top <count> candidates of a node
void print_candidate(node_t* parent, int count)
{
    if (parent->child_edge == NULL) return;
    edge_t* cur = parent->child_edge;
    node_t** cand = (node_t**)malloc(count * sizeof(node_t*));
    cand[0] = cur->to;
    int cnt = 0;
    while (cur != NULL) {
        if (cur->to->state.count > cand[0]->state.count) {
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
#define print_stat(i, st)                                                 \
    log("(%hhd, %hhd) => win: %.2lf%%, count: %.1lf%% (%d), eval: %.3lf", \
        st.pos.x, st.pos.y, get_rate(st, parent->state.id == 1 ? 1 : -1), \
        percentage(st.count, parent->state.count), st.count,              \
        ucb_eval(parent, cand[i], st.id == 1 ? -1 : 1))
    for (int i = 0; i < cnt; i++) {
        if (cand[i] != NULL) {
            print_stat(i, cand[i]->state);
        }
    }
    free(cand);
#undef CAND_SIZE
#undef percentage
#undef get_rate
#undef print_stat
}

/// @brief select best child by count
node_t* count_select(node_t* parent)
{
    if (parent->child_edge == NULL) return parent;
    edge_t* cur = parent->child_edge;
    node_t* sel = cur->to;
    while (cur != NULL) {
        if (cur->to->state.count > sel->state.count) {
            sel = cur->to;
        }
        cur = cur->next;
    }
    return sel;
}

/// @brief select best child by ucb value
node_t* ucb_select(node_t* parent)
{
    edge_t* cur = parent->child_edge;
    node_t* sel = cur->to;
    int flag = (parent->state.id == 1) ? 1 : -1;
    while (cur != NULL) {
        if (ucb_eval(parent, cur->to, flag) > ucb_eval(parent, sel, flag)) {
            sel = cur->to;
        }
        cur = cur->next;
    }
    return sel;
}

/// @brief check if state is terminated
bool terminated(state_t st)
{
    if (st.score || st.piece_cnt == st.capacity) return true;
    return false;
}

/// @brief put a piece at pos
/// @param parent current node
/// @param pos position of the piece
/// @return pointer to new node
node_t* put_piece(node_t* parent, point_t pos)
{
    int8_t i = pos.x, j = pos.y;
    // point_t new_begin = {
    //     max(min(parent->state.begin.x, i - mcts_parm.wrap_rad), 0),
    //     max(min(parent->state.begin.y, j - mcts_parm.wrap_rad), 0)};
    // point_t new_end = {
    //     min(max(parent->state.end.x, i + mcts_parm.wrap_rad + 1),
    //     BOARD_SIZE), min(max(parent->state.end.y, j + mcts_parm.wrap_rad +
    //     1), BOARD_SIZE)};
    add(parent->state.board, i, j, parent->state.id);
    state_t st = create_state(
        parent->state.board, parent->state.begin, parent->state.end,
        zobrist_update(parent->state.hash, pos, 0, parent->state.id), pos,
        parent->state.piece_cnt + 1, 3 - parent->state.id);
    minus(parent->state.board, i, j, parent->state.id);
    node_t* node = create_node(st);
    if (node == NULL) return NULL;
    append_child(parent, node);
    add(parent->state.visited, i, j, 1);
    parent->state.visited_cnt++;
    return node;
}

/// @brief if parent has a child at pos, return the child, else create a new one
node_t* preset_piece(node_t* parent, point_t pos)
{
    for (edge_t* edge = parent->child_edge; edge != NULL; edge = edge->next) {
        if (edge->to->state.pos.x == pos.x && edge->to->state.pos.y == pos.y) {
            return edge->to;
        }
    }
    return put_piece(parent, pos);
}

/// @brief traverse the tree to find a leaf node (terminate state), create new
/// nodes if needed
node_t* traverse(node_t* parent)
{
    // print_state(parent->state);
    // log("%d, %d, id = %d, score = %d", parent->state.pos.x,
    // parent->state.pos.y, parent->state.id, parent->state.score);
    if (parent == NULL || terminated(parent->state)) {
        // if (parent == NULL) return NULL;
        // board_t dec;
        // decode(parent->state.board, dec);
        // print(dec);
        // getchar();
        return parent;
    }
    // TODO: start
    point_t pp = parent->state.danger_pos[0];
    for (edge_t* e = parent->parent_edge; e != NULL; e = e->next) {
        node_t* grandparent = e->to;
        point_t p0 = grandparent->state.danger_pos[0],
                p1 = grandparent->state.danger_pos[1];
        if (inboard(p0) && get(parent->state.board, p0.x, p0.y) == 0) {
            // board_t dec;
            // decode(parent->state.board, dec);
            // print(dec);
            // logw("dangerous pos: (%d, %d)", parent->state.danger_pos.x,
            // parent->state.danger_pos.y); getchar(); log("exit");
            if (parent->state.id != first_id ||
                compressive_banned(parent->state.board, p0, first_id) ==
                    POS_ACCEPT) {
                return traverse(preset_piece(parent, p0));
            }
        }
        if (inboard(p1) && get(parent->state.board, p1.x, p1.y) == 0) {
            // log("exit");
            if (parent->state.id != first_id ||
                compressive_banned(parent->state.board, p1, first_id) ==
                    POS_ACCEPT) {
                return traverse(preset_piece(parent, p1));
            }
        }
    }
    if (inboard(pp)) {
        // log("exit");
        if (parent->state.id != first_id ||
            compressive_banned(parent->state.board, pp, first_id) ==
                POS_ACCEPT) {
            return traverse(preset_piece(parent, pp));
        }
    }
    // TODO: end
    // log("%d", parent->state.id);
    int res = parent->state.capacity - parent->state.visited_cnt;
    if (res) {
        int pos = (rand() % res) + 1;
        int8_t i = parent->state.begin.x, j = parent->state.begin.y;
        for (int t = 0, cnt = 0; t < parent->state.capacity; t++, j++) {
            if (j >= parent->state.end.y) i++, j = parent->state.begin.y;
            if (!get(parent->state.visited, i, j)) {
                cnt++;
                if (cnt >= pos) break;
            }
        }
        // log("res = %d, pos = %d, choose %d, %d", res, pos, i, j);
        point_t p = {i, j};
        return traverse(put_piece(parent, p));
    } else {
        if (parent->child_cnt == 0) {
            return parent;
        }
        return traverse(ucb_select(parent));
    }
}

// TODO: topo optimize
/// @brief backpropagate the score of a node to its ancestors
void backpropagate(node_t* node, int score, node_t* end)
{
    node->state.result += score;
    node->state.count++;
    if (node == end) return;
    for (edge_t* e = node->parent_edge; e != NULL; e = e->next) {
        backpropagate(e->to, score, end);
    }
}

/// @brief mcts main function
/// @param game current game info
/// @param player_assets player's assets
point_t mcts(const game_t game, mcts_parm_t parm)
{
    mcts_parm = parm;
    if (node_buffer == NULL)
        node_buffer = (node_t*)malloc(MAX_TREE_SIZE * sizeof(node_t));
    if (edge_buffer == NULL)
        edge_buffer = (edge_t*)malloc(MAX_TREE_SIZE * sizeof(edge_t) * 5);
    if (hashmap_buffer == NULL)
        hashmap_buffer =
            (hash_entry_t*)malloc(MAX_TREE_SIZE * sizeof(hash_entry_t));

    srand((unsigned)time(0));
    int start_time = record_time();

    tot = edge_tot = reused_tot = hashmap_tot = 0;
    hashmap = create_hashmap();
    first_id = game.first_player;

    int id = game.cur_player;
    cprboard_t zip;
    memset(zip, 0, sizeof(zip));

    if (game.count == 0) {
        point_t pos = {BOARD_SIZE / 2, BOARD_SIZE / 2};
        return pos;
    }

    point_t wrap_begin, wrap_end;
    int8_t radius = parm.wrap_rad;
    do {
        wrap_area(game.board, &wrap_begin, &wrap_end, radius);
    } while (((wrap_end.y - wrap_begin.y) * (wrap_end.x - wrap_begin.x) < 40) &&
             ++radius);

    node_t* root = create_node(create_state(
        zip, wrap_begin, wrap_end, 0, (point_t){0, 0}, 0, game.first_player));
    node_t* pre = NULL;
    for (int i = 0; i < game.count; i++) {
        pre = root;
        root = put_piece(pre, game.steps[i]);
    }

    int tim, cnt = 0;
    log("searching... (C: %.1lf->%.1lf, time: %d-%d, count: %d, rad: %d)",
        parm.start_c, parm.end_c, parm.min_time, parm.max_time, parm.min_count,
        radius);
    // int base = 1;
    int wanted_count = parm.min_count * root->state.capacity;
    while ((tim = get_time(start_time)) < parm.min_time ||
           (count_select(root)->state.count < wanted_count)) {
        if (tim > parm.max_time || tot >= NODE_LIMIT) break;
        mcts_parm.C = parm.start_c +
                      (parm.end_c - parm.start_c) * (double)tim / parm.max_time;
        node_t *leaf, *start = root;
        leaf = traverse(start);
        if (leaf != NULL) backpropagate(leaf, leaf->state.score, root);
        cnt++;
        // if (cnt % base == 0) {
        //     log("tried %d times.", cnt);
        //     base *= 2;
        // }
    }
    log("all count: %d, speed: %.2lf", cnt, (double)cnt / get_time(start_time));
    log("consumption: %d nodes (reused %d), %d edges, %d ms", tot, reused_tot,
        edge_tot, get_time(start_time));
    print_candidate(root, 7);
    node_t* move = count_select(root);
    free_hashmap(hashmap);
    log("clear cache");
    state_t st = move->state;
    int f = id == 1 ? 1 : -1;
    double rate = ((double)(st.count + (f)*st.result) / 2 / st.count * 100);
    if (rate < 80 && rate > 20)
        log("(%d, %d) => win: %.2lf%%, count: %.2lf%% (%d).", st.pos.x,
            st.pos.y, rate, (double)st.count / root->state.count * 100,
            st.count);
    else
        log_w("(%d, %d) => win: %.2lf%%, count: %.2lf%% (%d).", st.pos.x,
              st.pos.y, rate, (double)st.count / root->state.count * 100,
              st.count);
    return st.pos;
}