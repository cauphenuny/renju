// author: Cauphenuny <https://cauphenuny.github.io/>
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

long long tim_;

long long get_raw_time() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    long long tim = time.tv_sec * 1000 + time.tv_nsec / 1000000;
    return tim;
}

void reset_time() {
    tim_ = get_raw_time();
}

int get_time() {
    return (int)(get_raw_time() - tim_);
}

/****************************  manual ****************************/

int parse(char s[]) {
    if (isdigit(s[0])) return s[0] - '0';
    if (isupper(s[0])) return s[0] - 'A' + 10;
    if (islower(s[0])) return s[0] - 'a' + 10;
    return -1;
}

point_t manual(const board_t board) {
    point_t pos;
    char input_x[2], input_y[2];
    do {
        log("waiting input.");
        scanf("%s %s", input_x, input_y);
        pos.x = parse(input_x), pos.y = parse(input_y);
    } while ((!inboard(pos) || board[pos.x][pos.y]) && loge("invalid input!"));
    return pos;
}

/****************************  mcts ****************************/

const int CAPACITY = sizeof(board_t) / sizeof(int);

#ifndef TIME_LIMIT
    #define TIME_LIMIT 5000
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

state_t create_state(uint32_t *board, point_t pos) {
    state_t st;
    memset(&st, 0, sizeof(state_t));
    memcpy(st.board, board, sizeof(st.board));
    memcpy(st.visited, board, sizeof(st.board));
    board_t dec_board;
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

double C, D;
int E;

#undef log
double ucb_eval(node_t* node, int flag) {
    int win_cnt = node->state.count + flag * node->state.result;
    double f1 = (double)win_cnt / node->state.count;
    double f2 = sqrt(log(node->parent->state.count) / node->state.count);
    return f1 + C * f2;
}
#define log logi
double mix_eval(node_t* node, int flag) {
    double f1 = (double)node->state.result * flag / node->state.count;
    double f2 = (double)node->state.count / node->parent->state.count;
    return f1 + D * f2;
}

node_t* mix_select(node_t* parent) {
    if (parent->son == NULL) return parent;
    node_t *cur = parent->son, *sel = cur;
    int flag = (parent->state.id == 1) ? 1 : -1;
    while (cur != NULL) {
        if (mix_eval(cur, flag) > mix_eval(sel, flag)) {
            sel = cur;
        }
        cur = cur->next;
    }
    return sel;
}

node_t* count_select(node_t* parent) {
    if (parent->son == NULL) return parent;
    node_t *cur = parent->son, *sel = cur;
    //int flag = (parent->state.id == 1) ? 1 : -1;
    while (cur != NULL) {
        if (cur->state.count > sel->state.count) {
            sel = cur;
        }
        cur = cur->next;
    }
    return sel;
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

node_t* traverse(node_t* parent) {
    if (terminated(parent->state)) {
        //board_t b;
        //decode(parent->state.board, b);
        //print(b);
        return parent;
    }
    //log("%d\n", parent->state.id);
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
        set(parent->state.board, i, j, parent->state.id);
        state_t st = create_state(parent->state.board, p);
        set(parent->state.board, i, j, 0);
        node_t* node = create_node(st);
        if (node == NULL) return NULL;
        append_child(parent, node);
        set(parent->state.visited, i, j, 1);
        parent->state.visited_cnt++;
        return traverse(node);
    } else {
        return traverse(ucb_select(parent));
    }
}

void backpropagate(node_t* node, int score) {
    node->state.result += score;
    node->state.count++;
    if (node->parent != NULL) {
        backpropagate(node->parent, score);
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

point_t mcts(const board_t board, int id) {
    reset_time();
    log("start searching.");
    uint32_t zip[BOARD_SIZE];
    encode(board, zip);
    point_t p = {0, 0};
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (board[i][j] == 3 - id) {
                p.x = i, p.y = j;
            }
        }
    }
    node_t *root = create_node(create_state(zip, p));
    if (root->state.piece_cnt == 0) return (point_t){BOARD_SIZE / 2, BOARD_SIZE / 2};
    int cnt = 0, tim;
    double keyframe[5] = {0, 0.7, 0.9, 0.95, 0.97};
    while ((tim = get_time()) < TIME_LIMIT) {
    //while (count_select(root)->state.count < 10000) {
        cnt++;
        node_t *leaf, *start = root;
        if (cnt % E == 0) {
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
    //int size = delete_tree(root);
    double rate;
    if (id == 1) rate = (double)(st.count + st.result) / 2 / st.count * 100;
    else         rate = (double)(st.count - st.result) / 2 / st.count * 100;
    log("consumption: %d ms, %d nodes.", get_time(), tot);
    log("result: %.2lf%% (%d).", rate, st.count);
    tot = 0;
    return st.pos;
}

/****************************  export ****************************/

point_t player1(const board_t board) {
    C = 1.414, D = 100, E = 100;
    return mcts(board, 1);
    //return manual(board);
}

point_t player2(const board_t board) {
    //C = 1.414, D = 100, E = 200;
    //return mcts(board, 2);
    return manual(board);
}

void players_init() {
    srand((unsigned)time(0));
    memory_pool = (node_t*)malloc(MAX_TREE_SIZE * sizeof(node_t));
    //base_init();
}
