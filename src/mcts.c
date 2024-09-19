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

/**************************** mcts ****************************/

node_t* node_buffer;
edge_t* edge_buffer;

mcts_assets_t assets;

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

#define get(arr, x, y) (int)((arr[x] >> ((y) * 2)) & 3)
// #define get1(arr, x, y) ((arr[x] >> (y)) & 1)
// #define set(arr, x, y, v) (arr[x] += (((v - get(arr, x, y)) << ((y) * 2))))
#define add(arr, x, y, v)   arr[x] += ((v) << ((y) * 2))
#define minus(arr, x, y, v) arr[x] -= ((v) << ((y) * 2))

#define HASH_MAP_SIZE 10000019

hash_map_t create_hash_map()
{
    hash_map_t map;
    map.table = (hash_entry_t**)calloc(HASH_MAP_SIZE, sizeof(hash_entry_t*));
    return map;
}

void free_hash_map(hash_map_t map)
{
    // for (int i = 0; i < HASH_MAP_SIZE; i++) {
    //     hash_entry_t* entry = map.table[i];
    //     while (entry) {
    //         hash_entry_t* temp = entry;
    //         entry = entry->next;
    //         free(temp);
    //     }
    // }
    free(map.table);
}

unsigned int hash_function(zobrist_t key)
{
    return key % HASH_MAP_SIZE;
}

hash_entry_t* hash_buffer;

void hash_map_insert(hash_map_t map, zobrist_t key, node_t* value)
{
    unsigned int index = hash_function(key);
    // hash_entry_t* new_entry = (hash_entry_t*)malloc(sizeof(hash_entry_t));
    hash_entry_t* new_entry = hash_buffer + assets.hash_tot++;
    new_entry->key = key;
    new_entry->value = value;
    new_entry->next = map.table[index];
    map.table[index] = new_entry;
}

node_t* hash_map_get(hash_map_t map, zobrist_t key)
{
    unsigned int index = hash_function(key);
    hash_entry_t* entry = map.table[index];
    while (entry) {
        if (entry->key == key) {
            return entry->value;
        }
        entry = entry->next;
    }
    return NULL;
}

void hash_map_remove(hash_map_t map, zobrist_t key)
{
    unsigned int index = hash_function(key);
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

void print_state(state_t st)
{
    board_t b;
    decode(st.board, b);
    print(b);
}

int compressive_check(const cprboard_t bd, point_t pos)
{
    int id = get(bd, pos.x, pos.y);
    if (!id) return 0;
    static const int arrows[4][2] = {{0, 1}, {1, 0}, {1, 1}, {-1, 1}};
    for (int i = 0, a, b, cnt; i < 4; i++) {
        a = arrows[i][0], b = arrows[i][1];
        point_t np = {pos.x, pos.y};
        for (cnt = 0; inboard(np); np.x += a, np.y += b) {
            if (get(bd, np.x, np.y) == id)
                cnt++;
            else
                break;
        }
        np = (point_t){pos.x - a, pos.y - b};
        for (; inboard(np); np.x -= a, np.y -= b) {
            if (get(bd, np.x, np.y) == id)
                cnt++;
            else
                break;
        }
        if (cnt >= WIN_LENGTH) return id == 1 ? 1 : -1;
    }
    return 0;
}

int compressive_banned(const cprboard_t board, point_t pos, int id)
{
    return POS_ACCEPT;
    // print(board);
    // log("pos: (%d, %d), id: %d", pos.x, pos.y, id);
    if (id == -1) return POS_ACCEPT;
    cprboard_t bd;
    memcpy(bd, board, sizeof(cprboard_t));
    static const int arrows[4][2] = {{0, 1}, {1, 0}, {1, 1}, {-1, 1}};
    int cnt[8];
    for (int i = 0, a, b; i < 8; i++) {
        if (i < 4)
            a = arrows[i][0], b = arrows[i][1];
        else
            a = -arrows[i - 4][0], b = -arrows[i - 4][1];
        point_t np = {pos.x + a, pos.y + b};
        for (cnt[i] = 0; inboard(np); np.x += a, np.y += b) {
            if (get(bd, np.x, np.y) == id) {
                cnt[i]++;
            } else {
                break;
            }
        }
    }
    for (int i = 0; i < 4; i++) {
        // log("%c: %d/%d", "hv/\\"[i], cnt[i], cnt[i + 4]);
        if (cnt[i] + cnt[i + 4] + 1 >= 6) {
            return POS_BANNED_LONG;
        }
    }
    for (int i = 0; i < 4; i++) {
        if (cnt[i] + cnt[i + 4] + 1 == 5) {
            return POS_ACCEPT;
        }
    }
    minus(bd, pos.x, pos.y, get(bd, pos.x, pos.y));
    add(bd, pos.x, pos.y, id);
    // bd[pos.x][pos.y] = id;
    static const int live3s[3][10] = {
        {5, 0, 1, 1, 1, 0},
        {6, 0, 1, 0, 1, 1, 0},
        {6, 0, 1, 1, 0, 1, 0},
    };
    static const int exist4s[4][10] = {
        {4, 1, 1, 1, 1},
        {5, 1, 0, 1, 1, 1},
        {5, 1, 1, 1, 0, 1},
        {5, 1, 1, 0, 1, 1},
    };
    int live3_cnt = 0, exist4_cnt = 0;
    for (int i = 0, a, b; i < 4; i++) {
        a = arrows[i][0], b = arrows[i][1];
        for (int offset = -6; offset <= -1; offset++) {
            for (int j = 0; j < 3; j++) {
                const int* live3 = live3s[j];
                int n = live3[0];
                point_t np;
                np.x = pos.x + a * offset;
                np.y = pos.y + b * offset;
                for (int k = 1; inboard(np) && k <= n;
                     np.x += a, np.y += b, k++) {
                    // log("i=%d,j=%d,k=%d", i, j, k);
                    if (get(bd, np.x, np.y) == (live3[k] ? id : 0)) {
                        if (k == n) {
                            live3_cnt++;
                            offset += n - 1;
                        }
                    } else {
                        break;
                    }
                }
            }
        }
        for (int offset = -5; offset <= -1; offset++) {
            for (int j = 0; j < 4; j++) {
                const int* exist4 = exist4s[j];
                int n = exist4[0];
                point_t np;
                np.x = pos.x + a * offset;
                np.y = pos.y + b * offset;
                for (int k = 1; inboard(np) && k <= n;
                     np.x += a, np.y += b, k++) {
                    if (get(bd, np.x, np.y) == (exist4[k] ? id : 0)) {
                        if (k == n) {
                            exist4_cnt++;
                            // logw("arrow: %d(%d, %d)", i, a, b);
                            offset += (n - 1);
                        }
                    } else {
                        break;
                    }
                }
            }
        }
    }
    // if (live3_cnt || exist4_cnt) logw("%d, %d", live3_cnt, exist4_cnt);
    // prompt_getch();
    // bd[pos.x][pos.y] = 0;
    minus(bd, pos.x, pos.y, get(bd, pos.x, pos.y));
    if (live3_cnt > 1) {
        return POS_BANNED_33;
    }
    if (exist4_cnt > 1) {
        return POS_BANNED_44;
    }
    return POS_ACCEPT;
}

#define compressive_banned(...) POS_ACCEPT

void get_danger_pos(state_t* st, point_t pos)
{
    cpr_t* bd = st->board;
    int id = get(bd, pos.x, pos.y);
    // st->danger_pos[0] = st->danger_pos[1] = (point_t){-1, -1};
    if (!id) {
        return;
    }
    static const int arrows[4][2] = {{0, 1}, {1, 0}, {1, 1}, {-1, 1}};
    point_t np = pos, d0 = {-1, -1}, d1 = {-1, -1};
    // log("start.");
    for (int i = 0, a, b, cnt0 = 0, cnt1 = 0; i < 4; i++) {
        a = arrows[i][0], b = arrows[i][1];
        np = (point_t){pos.x, pos.y};
        for (int flag = cnt0 = cnt1 = 0; inboard(np) && flag < 2;
             np.x += a, np.y += b) {
            if (get(bd, np.x, np.y) == 3 - id) break;
            if (get(bd, np.x, np.y) == 0) {
                if (id == assets.first_id &&
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
        np = (point_t){pos.x - a, pos.y - b};
        for (int flag = 0; inboard(np) && flag < 2; np.x -= a, np.y -= b) {
            if (get(bd, np.x, np.y) == 3 - id) break;
            if (get(bd, np.x, np.y) == 0) {
                if (id == assets.first_id &&
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

state_t create_state(cprboard_t board, zobrist_t hash, point_t pos, int cnt)
{
    state_t st;
    memset(&st, 0, sizeof(state_t));
    memcpy(st.board, board, sizeof(st.board));
    memcpy(st.visited, board, sizeof(st.board));
    st.piece_cnt = cnt;
    st.visited_cnt = st.piece_cnt;
    st.id = 3 - get(board, pos.x, pos.y);
    st.hash = hash;
    // for (int i = assets.top; i < assets.bottom; i++) {
    //     for (int j = assets.left; j < assets.right; j++) {
    //         if (st.id == assets.first_id && !get(board, i, j) &&
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

node_t* create_node(state_t state)
{
    if (assets.tot >= NODE_LIMIT) return NULL;
    node_t* node;
    if ((node = hash_map_get(assets.hash_map, state.hash)) != NULL) {
        // exit(0);
        // board_t bd;
        // decode(state.board, bd);
        // log("state"), print(bd);
        // decode(node->state.board, bd);
        // log("node->state"), print(bd);
        // getchar();
        assets.reused_tot++;
        node->state.pos = state.pos;
        // get_danger_pos(&node->state, state.pos);
        return node;
    }

    // node = (node_t*)malloc(sizeof(node_t));
    node = node_buffer + assets.tot++;
    memset(node, 0, sizeof(node_t));
    memcpy(&node->state, &state, sizeof(state_t));
    hash_map_insert(assets.hash_map, state.hash, node);
    return node;
}

int append_child(node_t* parent, node_t* node)
{
    edge_t* son_edge = edge_buffer + assets.edge_tot++;
    edge_t* parent_edge = edge_buffer + assets.edge_tot++;
    // edge_t* son_edge = (edge_t*)malloc(sizeof(edge_t));
    // edge_t* parent_edge = (edge_t*)malloc(sizeof(edge_t));
    memset(son_edge, 0, sizeof(edge_t));
    memset(parent_edge, 0, sizeof(edge_t));

    son_edge->next = parent->son_edge;
    son_edge->to = node;
    parent->son_edge = son_edge;
    parent->son_cnt++;

    parent_edge->next = node->parent_edge;
    parent_edge->to = parent;
    node->parent_edge = parent_edge;
    node->parent_cnt++;

    return parent->son_cnt;
}

int delete_subgraph(node_t* node)
{
    int size = 1;
    node_t* child;
    for (edge_t *e = node->son_edge, *nxt; e != NULL; e = nxt) {
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
    hash_map_remove(assets.hash_map, node->state.hash);
    free(node);
    return size;
}

#undef log
double ucb_eval(node_t* parent, node_t* node, int flag)
{
    int win_cnt = node->state.count + flag * node->state.result;
    double f1 = (double)win_cnt / node->state.count;
    double f2 = sqrt(log(parent->state.count) / node->state.count);
    return f1 + assets.mcts_parm.C * f2;
}
#define log logi

void print_candidate(node_t* parent, int count)
{
    if (parent->son_edge == NULL) return;
    edge_t* cur = parent->son_edge;
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
#define get_rate(st, f)  (percentage((st.count + (f) * st.result) / 2, st.count))
#define print_stat(i, st)                                                   \
    log("#%d: (%d, %d) => win: %.2lf%%, count: %.2lf%% (%d).", i, st.pos.x, \
        st.pos.y, get_rate(st, parent->state.id == 1 ? 1 : -1),             \
        percentage(st.count, parent->state.count), st.count)
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

node_t* count_select(node_t* parent)
{
    if (parent->son_edge == NULL) return parent;
    edge_t* cur = parent->son_edge;
    node_t* sel = cur->to;
    while (cur != NULL) {
        if (cur->to->state.count > sel->state.count) {
            sel = cur->to;
        }
        cur = cur->next;
    }
    return sel;
}

node_t* ucb_select(node_t* parent)
{
    edge_t* cur = parent->son_edge;
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

bool terminated(state_t st)
{
    if (st.score || st.piece_cnt == assets.capacity) return true;
    return false;
}

node_t* put_piece(node_t* parent, point_t pos)
{
    int i = pos.x, j = pos.y;
    add(parent->state.board, i, j, parent->state.id);
    state_t st = create_state(
        parent->state.board,
        zobrist_update(parent->state.hash, pos, 0, parent->state.id), pos,
        parent->state.piece_cnt + 1);
    minus(parent->state.board, i, j, parent->state.id);
    node_t* node = create_node(st);
    if (node == NULL) return NULL;
    append_child(parent, node);
    add(parent->state.visited, i, j, 1);
    parent->state.visited_cnt++;
    return node;
}

node_t* preset_piece(node_t* parent, point_t pos)
{
    for (edge_t* edge = parent->son_edge; edge != NULL; edge = edge->next) {
        if (edge->to->state.pos.x == pos.x && edge->to->state.pos.y == pos.y) {
            return edge->to;
        }
    }
    return put_piece(parent, pos);
}

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
            if (parent->state.id != assets.first_id ||
                compressive_banned(parent->state.board, p0, first_id) ==
                    POS_ACCEPT) {
                return traverse(preset_piece(parent, p0));
            }
        }
        if (inboard(p1) && get(parent->state.board, p1.x, p1.y) == 0) {
            // log("exit");
            if (parent->state.id != assets.first_id ||
                compressive_banned(parent->state.board, p1, first_id) ==
                    POS_ACCEPT) {
                return traverse(preset_piece(parent, p1));
            }
        }
    }
    if (inboard(pp)) {
        // log("exit");
        if (parent->state.id != assets.first_id ||
            compressive_banned(parent->state.board, pp, first_id) ==
                POS_ACCEPT) {
            return traverse(preset_piece(parent, pp));
        }
    }
    // TODO: end
    // log("%d", parent->state.id);
    int res = assets.capacity - parent->state.visited_cnt;
    if (res) {
        int pos = (rand() % res) + 1;
        int i = assets.top, j = assets.left;
        for (int t = 0, cnt = 0; t < assets.capacity; t++, j++) {
            if (j >= assets.right) i++, j = assets.left;
            if (!get(parent->state.visited, i, j)) {
                cnt++;
                if (cnt >= pos) break;
            }
        }
        // log("res = %d, pos = %d, choose %d, %d", res, pos, i, j);
        point_t p = {i, j};
        return traverse(put_piece(parent, p));
    } else {
        if (parent->son_cnt == 0) {
            return parent;
        }
        return traverse(ucb_select(parent));
    }
}

// TODO: topo optimize
void backpropagate(node_t* node, int score)
{
    node->state.result += score;
    node->state.count++;
    for (edge_t* e = node->parent_edge; e != NULL; e = e->next) {
        backpropagate(e->to, score);
    }
}

// void test(board_t board) {
//     state_t st;
//     board_t ans = {0};
//     //logw("ans: %p, board: %p\n", (void*)ans, (void*)board);
//     log("start encode");
//     encode(board, st.board);
//     log("start decode");
//     decode(st.board, ans);
//     log("start print");
//     print(ans);
//     log("#1 printed");
//     for (int i = 0; i < BOARD_SIZE; i++) {
//         for (int j = 0; j < BOARD_SIZE; j++) {
//             ans[i][j] = get(st.board, i, j);
//         }
//     }
//     print(ans);
//     log("#2 printed");
// }

point_t mcts(const game_t game, mcts_assets_t* player_assets)
{
    if (node_buffer == NULL)
        node_buffer = (node_t*)malloc(MAX_TREE_SIZE * sizeof(node_t));
    if (edge_buffer == NULL)
        edge_buffer = (edge_t*)malloc(MAX_TREE_SIZE * sizeof(edge_t) * 5);
    if (hash_buffer == NULL)
        hash_buffer =
            (hash_entry_t*)malloc(MAX_TREE_SIZE * sizeof(hash_entry_t));

    srand((unsigned)time(0));
    reset_time();

    assets = *player_assets;
    assets.tot = assets.edge_tot = assets.reused_tot = assets.hash_tot = 0;
    assets.hash_map = create_hash_map();
    assets.first_id = game.first_id;

    mcts_parm_t parm = assets.mcts_parm;
    board_t board;
    memcpy(&board, game.board, sizeof(board_t));
    int id = game.current_id;
    cprboard_t zip;
    encode(game.board, zip);
    if (game.step_cnt == 0) {
        point_t pos = {BOARD_SIZE / 2, BOARD_SIZE / 2};
        add(zip, pos.x, pos.y, id);
        assets.last_status =
            create_state(zip, zobrist_update(0, pos, 0, id), pos, 1);
        (*player_assets) = assets;
        return pos;
    }
    point_t p0 = game.steps[game.step_cnt - 1];
    node_t *root, *pre;
    if (assets.last_status.piece_cnt == 0) {
        root = create_node(create_state(zip, zobrist_update(0, p0, 0, 3 - id),
                                        p0, game.step_cnt));
        pre = NULL;
    } else {
        pre = create_node(assets.last_status);
        root = put_piece(pre, p0);
        append_child(pre, root);
    }
    // board_t bd;
    // decode(assets.root->state.board, bd);
    // log("root"), print(bd);
    // decode(prev_root->state.board, bd);
    // log("prev_root"), print(bd);
    // root = create_node(create_state(zip, p0, game.step_cnt));
    get_danger_pos(&(root->state), p0);
    root->state.score = 0;
    // if (game.step_cnt > 1) {
    //     point_t p1 = game.steps[game.step_cnt - 2];
    //     virtual_root = create_node(create_state(zip, p1, game.step_cnt - 1));
    //     get_danger_pos(&(virtual_root->state), p1);
    //     append_child(virtual_root, root);
    // }
    do {
        wrap_area(board, &assets.top, &assets.bottom, &assets.left,
                  &assets.right, parm.WRAP_RAD);
        assets.capacity =
            (assets.right - assets.left) * (assets.bottom - assets.top);
    } while ((assets.capacity < 50) && (parm.WRAP_RAD++));
    // log("capacity: %d", capacity);
    // log("virtual_root's danger pos: (%d, %d)/(%d, %d)",
    // virtual_root->state.danger_pos[0].x, virtual_root->state.danger_pos[0].y,
    // virtual_root->state.danger_pos[1].x,
    // virtual_root->state.danger_pos[1].y); log("root's danger pos: (%d,
    // %d)/(%d, %d)", root->state.danger_pos[0].x, root->state.danger_pos[0].y,
    // root->state.danger_pos[1].x, root->state.danger_pos[1].y); log("score:
    // %d, cnt: %d", root->state.score, root->state.piece_cnt);
    int tim, cnt = 0;
    // double keyframe[5] = {0, 0.8, 0.9, 0.95}; // TODO
    log("searching... (C: %.2lf, time: %d-%d, count: %d, rad: %d)", parm.C,
        parm.MIN_TIME, parm.MAX_TIME, parm.MIN_COUNT, parm.WRAP_RAD);
    // while ((tim = get_time()) < parm.TIME_LIMIT &&
    int base = 4096;
    // while ((tim = get_time()) < parm.TIME_LIMIT ) {
    while ((tim = get_time()) < parm.MIN_TIME ||
           ((tim < parm.MAX_TIME) && (assets.tot < NODE_LIMIT) &&
            (count_select(root)->state.count < parm.MIN_COUNT * assets.capacity))) {
        node_t *leaf, *start = root;
        leaf = traverse(start);
        if (leaf != NULL) backpropagate(leaf, leaf->state.score);
        cnt++;
        if (cnt % base == 0) {
            //log("tried %d times.", cnt);
            base *= 2;
        }
    }
    log("all count: %d, speed: %.2lf", root->state.count,
        (double)root->state.count / get_time());
    log("consumption: %d nodes (reused %d), %d edges, %d ms", assets.tot,
        assets.reused_tot, assets.edge_tot, get_time());
    print_candidate(root, 7);
    node_t* move = count_select(root);
    free_hash_map(assets.hash_map);
    log("clear cache");
    // for (edge_t *edge = root->son_edge, *bro; edge != NULL; edge = bro) {
    //     bro = edge->next;
    //     if (edge->to != move) {
    //         assets.tot -= delete_subgraph(edge->to);
    //     } else {
    //         root->son_edge = edge, edge->next = NULL;
    //     }
    // }
    assets.last_status = move->state;
    state_t st = move->state;
    point_t pos = st.pos;
    int f = id == 1 ? 1 : -1;
    double rate = ((double)(st.count + (f)*st.result) / 2 / st.count * 100);
    log("all consumption: %d ms", get_time());
    if (rate < 80 && rate > 20)
        log("(%d, %d) => win: %.2lf%%, count: %.2lf%% (%d).", st.pos.x,
            st.pos.y, rate, (double)st.count / root->state.count * 100,
            st.count);
    else
        logw("(%d, %d) => win: %.2lf%%, count: %.2lf%% (%d).", st.pos.x,
             st.pos.y, rate, (double)st.count / root->state.count * 100,
             st.count);
    (*player_assets) = assets;
    return pos;
}

// void assets_init(mcts_assets_t* ast)
// {
//     // ast->hash_map = create_hash_map();
// }