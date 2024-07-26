// author: Cauphenuny <https://cauphenuny.github.io/>
// date: 2024/07/24
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>

#include "util.h"
#include "board.h"
#include "server.h"
#include "mcts.h"


/**************************** mcts ****************************/

const int CAPACITY = sizeof(board_t) / sizeof(int);

#ifndef TIME_LIMIT
    #define TIME_LIMIT 3000
#endif

void encode(const board_t src, uint32_t dest[]) {
    for (int i = 0; i < BOARD_SIZE; i++) {
        dest[i] = 0;
        for (int j = BOARD_SIZE - 1; j >= 0; j--) {
            dest[i] = dest[i] * 4 + src[i][j];
            if (src[i][j] > 2) dest[i] -= 2;
        }
    }
}

void decode(const uint32_t src[], board_t dest) {
    for (int i = 0; i < BOARD_SIZE; i++) {
        uint32_t tmp = src[i];
        for (int j = 0; j < BOARD_SIZE; j++) {
            dest[i][j] = tmp & 3;
            tmp >>= 2;
        }
    }
}

typedef struct {
    uint32_t board[BOARD_SIZE];
    uint32_t visited[BOARD_SIZE];
    int piece_cnt;
    int visited_cnt;
    int result;
    int count;
    int8_t id;
    int8_t score;
    point_t pos;
    point_t danger_pos[2];
} state_t;

#define get(arr, x, y) ((arr[x] >> ((y) * 2)) & 3)
#define set(arr, x, y, v) (arr[x] += (((v - get(arr, x, y)) << ((y) * 2))))

typedef struct node_t {
    state_t state;
    struct node_t* parent;
    struct node_t* son;
    int son_cnt;
    struct node_t* next;
} node_t;

#define MAX_TREE_SIZE 20001000
#define NODE_LIMIT 20000000

node_t *memory_pool;
int tot;

bool __get_danger_pos_log_flag;

void print_state(state_t st) {
    board_t b;
    decode(st.board, b);
    print(b);
}

void get_danger_pos(state_t* st, point_t pos) {
    uint32_t* bd = st->board;
    int id = get(bd, pos.x, pos.y);
    //st->danger_pos[0] = st->danger_pos[1] = (point_t){-1, -1};
    if (!id) { return; }
    static const int arrows[4][2] = {{0, 1}, {1, 0}, {1, 1}, {-1, 1}};
    point_t np = pos, d0 = {-1, -1}, d1 = {-1, -1};
    //log("start.");
    for (int i = 0, a, b, cnt0 = 0, cnt1 = 0; i < 4; i++) {
        a = arrows[i][0], b = arrows[i][1];
        np = (point_t){pos.x, pos.y};
        for (int flag = cnt0 = cnt1 = 0; inboard(np) && flag < 2; np.x += a, np.y += b) {
            if (get(bd, np.x, np.y) == 3 - id) break;
            if (get(bd, np.x, np.y) == 0) { flag++; if (flag == 1) d1 = np; }
            if (flag == 0) { cnt0++; cnt1++; }
            if (flag == 1) { cnt1++; }
        }
        //log("cnt0: %d, cnt1: %d", cnt0, cnt1);
        np = (point_t){pos.x - a, pos.y - b};
        for (int flag = 0; inboard(np) && flag < 2; np.x -= a, np.y -= b) {
            if (get(bd, np.x, np.y) == 3 - id) break;
            if (get(bd, np.x, np.y) == 0) { flag++; if (flag == 1) d0 = np; }
            if (flag == 0) { cnt0++; cnt1++; }
            if (flag == 1) { cnt0++; }
        }
        //log("cnt0: %d, cnt1: %d", cnt0, cnt1);
        if (inboard(d0) && cnt0 >= WIN_LENGTH) {
            if (st->danger_pos[0].x == -1) {
                st->danger_pos[0] = d0;
            } else {
                if (d0.x != st->danger_pos[0].x || d0.y != st->danger_pos[0].y) {
                    st->danger_pos[1] = d0;
                }
            }
            //if (__get_danger_pos_log_flag) {
            //    log("found danger pos0: (%d, %d) ---(%d, %d)--> (%d, %d), id=%d", pos.x, pos.y, -a, -b, d0.x, d0.y, id);
            //    log("now: (%d, %d), (%d, %d)", st->danger_pos[0].x, st->danger_pos[0].y, st->danger_pos[1].x, st->danger_pos[1].y);
            //    //getchar();
            //}
        }
        if (inboard(d1) && cnt1 >= WIN_LENGTH) {
            assert(inboard(d1));
            if (st->danger_pos[0].x == -1) {
                st->danger_pos[0] = d1;
            } else {
                if (d1.x != st->danger_pos[0].x || d1.y != st->danger_pos[0].y) {
                    st->danger_pos[1] = d1;
                }
            }
            //if (__get_danger_pos_log_flag) {
            //    log("found danger pos0: (%d, %d) ---(%d, %d)--> (%d, %d), id=%d", pos.x, pos.y, -a, -b, d1.x, d1.y, id);
            //    log("now: (%d, %d), (%d, %d)", st->danger_pos[0].x, st->danger_pos[0].y, st->danger_pos[1].x, st->danger_pos[1].y);
            //    //getchar();
            //}
        }
    }
    //if (__get_danger_pos_log_flag) {
    //    log("result: (%d, %d), (%d, %d)", st->danger_pos[0].x, st->danger_pos[0].y, st->danger_pos[1].x, st->danger_pos[1].y);
    //}
}

state_t create_state(uint32_t *board, point_t pos) {
    state_t st;
    memset(&st, 0, sizeof(state_t));
    memcpy(st.board, board, sizeof(st.board));
    memcpy(st.visited, board, sizeof(st.board));
    static board_t dec_board;
    decode(st.board, dec_board);
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (dec_board[i][j]) {
                st.piece_cnt++;
            }
        }
    }
    st.visited_cnt = st.piece_cnt;
    st.id = 3 - dec_board[pos.x][pos.y];
    st.score = check(dec_board, pos);
    //if(st.score) log("(%d, %d): %d", pos.x, pos.y, st.id);
    st.pos = pos;
    st.danger_pos[0] = st.danger_pos[1] = (point_t){-1, -1};
    get_danger_pos(&st, pos);
    //log("calc danger pos: (%d, %d), all: %d\n", st.danger_pos.x, st.danger_pos.y, count);
    //if (st->danger_pos > 1) {
    //    st.score = (st.id == 1) ? -1 : 1;
    //    //print(dec_board);
    //    //logw("early terminated, score = %d", st.score);
    //    //getchar();
    //}
    return st;
}

node_t* create_node(state_t state) {
    if (tot >= NODE_LIMIT) return NULL;
    node_t* node = memory_pool + tot;
    tot++;
    memset(node, 0, sizeof(node_t));
    memcpy(&node->state, &state, sizeof(state_t));
    return node;
}

int append_child(node_t* parent, node_t* node) {
    node->parent = parent;
    node->next = parent->son;
    parent->son = node;
    parent->son_cnt++;
    return parent->son_cnt;
}

int delete_tree(node_t* node) {
    node_t *son = node->son, *bro;
    int size = 1;
    while (son != NULL) {
        bro = son->next;
        size += delete_tree(son);
        son = bro;
    }
    free(node);
    return size;
}

mcts_parm_t mcts_parm;

#undef log
double ucb_eval(node_t* node, int flag) {
    int win_cnt = node->state.count + flag * node->state.result;
    double f1 = (double)win_cnt / node->state.count;
    double f2 = sqrt(log(node->parent->state.count) / node->state.count);
    return f1 + mcts_parm.C * f2;
}
#define log logi

node_t* count_select(node_t* parent) {
    if (parent->son == NULL) return parent;
    node_t *cur = parent->son, *sel = cur;
    //int flag = (parent->state.id == 1) ? 1 : -1;
#define CAND_SIZE 4
    //node_t* cand[CAND_SIZE] = {NULL, NULL, NULL, NULL};
    while (cur != NULL) {
        if (cur->state.count > sel->state.count) {
            //for (int i = CAND_SIZE - 1; i > 0; i--) {
            //    cand[i] = cand[i - 1];
            //}
            //cand[0] = sel;
            sel = cur;
        }
        cur = cur->next;
    }
#define percentage(a, b) ((double)(a) / (double)(b) * 100)
#define get_rate(st, f) (percentage((st.count + (f) * st.result) / 2 , st.count))
#define print_stat(st) \
    log("candidate: (%d, %d) => win: %.2lf%%, count: %.2lf%% (%d).", st.pos.x, st.pos.y, get_rate(st, parent->state.id == 1 ? 1 : -1), percentage(st.count, parent->state.count), st.count)
    //print_stat(sel->state);
    //for (int i = 0; i < CAND_SIZE; i++) {
    //    if (cand[i] != NULL) {
    //        print_stat(cand[i]->state);
    //    }
    //}
    return sel;
#undef CAND_SIZE
#undef percentage
#undef get_rate
#undef print_stat
}

node_t* ucb_select(node_t* parent) {
    node_t *cur = parent->son, *sel = cur;
    int flag = (parent->state.id == 1) ? 1 : -1;
    while (cur != NULL) {
        if (ucb_eval(cur, flag) > ucb_eval(sel, flag)) {
            sel = cur;
        }
        cur = cur->next;
    }
    return sel;
}

bool terminated(state_t st) {
    if (st.score || st.piece_cnt == CAPACITY) return true;
    return false;
}

node_t* put_piece(node_t* parent, point_t pos) {
    int i = pos.x, j = pos.y;
    set(parent->state.board, i, j, parent->state.id);
    state_t st = create_state(parent->state.board, pos);
    set(parent->state.board, i, j, 0);
    node_t* node = create_node(st);
    if (node == NULL) return NULL;
    append_child(parent, node);
    set(parent->state.visited, i, j, 1);
    parent->state.visited_cnt++;
    return node;
}

node_t* preset_piece(node_t* parent, point_t pos) {
    for (node_t* son = parent->son; son != NULL; son = son->next) {
        if (son->state.pos.x == pos.x && son->state.pos.y == pos.y) {
            return son;
        }
    }
    return put_piece(parent, pos);
}

node_t* traverse(node_t* parent) {
    //print_state(parent->state);
    //log("%d, %d, id = %d, score = %d", parent->state.pos.x, parent->state.pos.y, parent->state.id, parent->state.score);
    if (parent == NULL || terminated(parent->state)) {
        //if (parent == NULL) return NULL;
        //board_t dec;
        //decode(parent->state.board, dec);
        //print(dec);
        //getchar();
        return parent;
    }
    if (parent->parent != NULL) {
        point_t p0 = parent->parent->state.danger_pos[0], p1 = parent->parent->state.danger_pos[1];
        point_t pp = parent->state.danger_pos[0];
        if (inboard(p0) && get(parent->state.board, p0.x, p0.y) == 0) {
            //board_t dec;
            //decode(parent->state.board, dec);
            //print(dec);
            //logw("dangerous pos: (%d, %d)", parent->state.danger_pos.x, parent->state.danger_pos.y);
            //getchar();
            //log("exit");
            return traverse(preset_piece(parent, p0));
        }
        if (inboard(p1) && get(parent->state.board, p1.x, p1.y) == 0) {
            //log("exit");
            return traverse(preset_piece(parent, p1));
        }
        if (inboard(pp)) {
            //log("exit");
            return traverse(preset_piece(parent, pp));
        }
    }
    //log("%d", parent->state.id);
    int res = CAPACITY - parent->state.visited_cnt;
    if (res) {
        int pos = (rand() % res) + 1;
        int i = -1, j = -1;
        for (int t = 0, cnt = 0; cnt < pos && t < CAPACITY; t++) {
            i = t / BOARD_SIZE;
            j = t % BOARD_SIZE;
            if (!get(parent->state.visited, i, j)) {
                cnt++;
            }
        }
        //log("res = %d, pos = %d, choose %d, %d", res, pos, i, j);
        point_t p = {i, j};
        return traverse(put_piece(parent, p));
    } else {
        return traverse(ucb_select(parent));
    }
}

void backpropagate(node_t* node, int score) {
    while (node != NULL) {
        node->state.result += score;
        node->state.count++;
        node = node->parent;
    }
}

//void test(board_t board) {
//    state_t st;
//    board_t ans = {0};
//    //logw("ans: %p, board: %p\n", (void*)ans, (void*)board);
//    log("start encode");
//    encode(board, st.board);
//    log("start decode");
//    decode(st.board, ans);
//    log("start print");
//    print(ans);
//    log("#1 printed");
//    for (int i = 0; i < BOARD_SIZE; i++) {
//        for (int j = 0; j < BOARD_SIZE; j++) {
//            ans[i][j] = get(st.board, i, j);
//        }
//    }
//    print(ans);
//    log("#2 printed");
//}

point_t mcts(const board_t board, int id, mcts_parm_t parm) {
    log("initializing.");
    srand((unsigned)time(0));
    if (memory_pool == NULL) {
        memory_pool = (node_t*)malloc(MAX_TREE_SIZE * sizeof(node_t));
    }
    reset_time();
    mcts_parm = parm;
    node_t *virtual_root, *root;
  {
    uint32_t zip[BOARD_SIZE];
    encode(board, zip);
    point_t p0 = {0, 0}, p1 = {0, 0}, p;
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (board[i][j] == 3 - id) {
                p0.x = i, p0.y = j;
            } else if (board[i][j] == id) {
                p1.x = i, p1.y = j;
            }
        }
    }
    __get_danger_pos_log_flag = 1;
    root = create_node(create_state(zip, p0));
    virtual_root = create_node(create_state(zip, p1));
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (board[i][j]) {
                p.x = i, p.y = j;
                if (board[i][j] == id) {
                    get_danger_pos(&(virtual_root->state), p);
                } else {
                    get_danger_pos(&(root->state), p);
                }
            }
        }
    }
    root->state.score = 0;
    append_child(virtual_root, root);
    __get_danger_pos_log_flag = 0;
    if (root->state.piece_cnt == 0) return (point_t){BOARD_SIZE / 2, BOARD_SIZE / 2};
  }
    //log("virtual_root's danger pos: (%d, %d)/(%d, %d)", virtual_root->state.danger_pos[0].x, virtual_root->state.danger_pos[0].y, virtual_root->state.danger_pos[1].x, virtual_root->state.danger_pos[1].y);
    //log("root's danger pos: (%d, %d)/(%d, %d)", root->state.danger_pos[0].x, root->state.danger_pos[0].y, root->state.danger_pos[1].x, root->state.danger_pos[1].y);
    //log("score: %d, cnt: %d", root->state.score, root->state.piece_cnt);
  {
    int tim;
    int cnt = 0;
    double keyframe[5] = {0, 0.7, 0.9, 0.95, 0.97};
    log("start searching.");
    while ((tim = get_time()) < TIME_LIMIT) {
    //while (count_select(root)->state.count < 10000) {
        node_t *leaf, *start = root;
        cnt++;
        if (cnt % parm.M == 0) {
            for (int i = 1; i < 5; i++) {
                if (tim > TIME_LIMIT * keyframe[i]) {
                    start = count_select(start);
                } else {
                    break;
                }
            }
        }
        leaf = traverse(start);
        if (leaf != NULL) backpropagate(leaf, leaf->state.score);
    }
    node_t *move = count_select(root);
    state_t st = move->state;
    point_t pos = st.pos;
    //int size = delete_tree(root);
#define get_rate(st, f) ((double)(st.count + (f) * st.result) / 2 / st.count * 100)
    log("consumption: %d ms, %d nodes.", get_time(), tot);
    log("(%d, %d) => win: %.2lf%%, count: %.2lf%% (%d).", st.pos.x, st.pos.y, get_rate(st, id == 1 ? 1 : -1), (double)st.count / root->state.count * 100, st.count);
#undef get_rate
    tot = 0;
    return pos;
  }
}
