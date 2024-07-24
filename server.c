// author: Cauphenuny <https://cauphenuny.github.io/>
#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <signal.h>
#include "util.h"
#include "board.h"
#include "players.h"

char id2name(int id) {
    if (id < 10) {
        return id + '0';
    } else {
        return id - 10 + 'A';
    }
}

/*
   | 0 | 1 | 2 | 3 | 4 |
---+---+---+---+---+---+
 0 | o |   |   |   |   |
---+---+---+---+---+---+
*/

board_t board;

void wrap_area(const board_t bd, int *top, int *bottom, int *left, int *right, int radius) {
    int n = BOARD_SIZE, mid = n / 2;
    *top = *left = mid - radius;
    *bottom = *right = mid + radius + 1;
    chkmin(*right, n), chkmin(*bottom, n);
    chkmax(*top, 0), chkmax(*left, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (bd[i][j]) {
                chkmin(*top,  max(0, i - radius));
                chkmin(*left, max(0, j - radius));
                chkmax(*bottom, min(i + radius + 1, n));
                chkmax(*right,  min(j + radius + 1, n));
            }
        }
    }
}

void print(const board_t bd) {
    // 0: empty, 1/2: prev p1/p2 piece, 3/4: cur p1/p2 piece
    char* ch[5] = { 
        " ", 
        "\033[" CLI_COLOR_GREEN "mo\033[0m",
        "\033[" CLI_COLOR_RED "mx\033[0m",
        "\033[" CLI_STYLE_UNDERLINE "m\033[" CLI_STYLE_BOLD ";" CLI_COLOR_GREEN "mo\033[0m",
        "\033[" CLI_STYLE_UNDERLINE "m\033[" CLI_STYLE_BOLD ";" CLI_COLOR_RED "mx\033[0m",
    };
    int top, bottom, left, right;
    wrap_area(bd, &top, &bottom, &left, &right, 3);
    printf("   |");
    for (int i = left; i < right; i++) {
        printf(" %c |%c", id2name(i), "\n"[i != right - 1]);
    }
    for (int i = left; i < right + 1; i++) {
        printf("---+%c", "\n"[i != right]);
    }
    for (int i = top; i < bottom; i++) {
        printf(" %c |", id2name(i));
        for (int j = left; j < right; j++) {
            printf(" %s |%c", ch[bd[i][j]], "\n"[j != right - 1]);
        }
        for (int j = left; j < right + 1; j++) {
            printf("---+%c", "\n"[j != right]);
        }
    }
}

bool inboard(point_t pos) {
    int n = BOARD_SIZE;
    return pos.x >= 0 && pos.x < n && pos.y >= 0 && pos.y < n;
}

void refresh() {
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (board[i][j] > 2) board[i][j] -= 2;
        }
    }
}

bool put(int id, point_t pos) {
    if (!inboard(pos) || board[pos.x][pos.y]) return true;
    refresh();
    board[pos.x][pos.y] = id + 2;
    return false;
}

int check(board_t bd, point_t pos) {
    int id = bd[pos.x][pos.y];
    if (!id) return 0;
    int arrows[8][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}, {1, 1}, {-1, 1}, {1, -1}, {-1, -1}};
    for (int i = 0; i < 8; i++) {
        int a = arrows[i][0], b = arrows[i][1];
        for (int k = -WIN_LENGTH + 1; k <= 0; k++) {
            bool flag = true;
            for (int j = 0; j < WIN_LENGTH ; j++) {
                point_t np = {pos.x + (j + k) * a, pos.y + (j + k) * b};
                if (!inboard(np) || (bd[np.x][np.y] != id)) {
                    flag = 0;
                    break;
                }
            }
            if (flag) return id == 1 ? 1 : -1;
        }
    }
    return 0;
}

int game(bool reverse_flag) {
    players_init();
    memset(board, 0, sizeof(board));
    print(board);
    point_t pos;
    if (reverse_flag) {
        log("player2's turn.");
        while (put(2, pos = player2(board))) { loge("invalid position!"); }
        print(board);
        refresh();
    }
    while (1) {
        log("player1's turn.");
        while (put(1, pos = player1(board))) { loge("invalid position!"); }
        print(board);
        refresh();
        if (check(board, pos)) {
            log("player1 wins.");
            return 1;
        }
        log("player2's turn.");
        while (put(2, pos = player2(board))) { loge("invalid position!"); }
        print(board);
        refresh();
        if (check(board, pos)) {
            log("player2 wins.");
            return 2;
        }
        //test(board);
    }
}

int result[3];

void signal_handler(int signum) {
    log("received signal %d, terminate.", signum);
    int r1 = result[1], r2 = result[2];
    if (r1 + r2) {
        log("result: %d / %d (%.2lf%%).", r1, r2, (double)r1 / (r1 + r2) * 100);
    }
    exit(0);
}

int main() {
    signal(SIGINT, signal_handler);
    while (1) {
        result[game(0)]++;
        log("result: %d / %d", result[1], result[2]);
        result[game(1)]++;
        log("result: %d / %d", result[1], result[2]);
    }
    return 0;
}
