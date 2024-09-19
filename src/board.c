// author: Cauphenuny <https://cauphenuny.github.io/>
// date: 2024/07/27
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>

#include "util.h"
#include "board.h"

const char* POS_BAN_MSG[] = {
    "accept",
    "long banned",
    "33 banned",
    "44 banned",
};

/// @brief calculate the wrap area of chessboard: $[top, bottom) \times [left,
/// right)$
/// @param board chessboard
/// @param top return top
/// @param bottom return bottom
/// @param left return left
/// @param right return right
/// @param radius wrap radius
void wrap_area(const board_t board, int* top, int* bottom, int* left,
               int* right, int radius) {
    int n = BOARD_SIZE, mid = n / 2;
    *top = *left = mid - radius;
    *bottom = *right = mid + radius + 1;
    chkmin(*right, n), chkmin(*bottom, n);
    chkmax(*top, 0), chkmax(*left, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (board[i][j]) {
                chkmin(*top, max(0, i - radius));
                chkmin(*left, max(0, j - radius));
                chkmax(*bottom, min(i + radius + 1, n));
                chkmax(*right, min(j + radius + 1, n));
            }
        }
    }
}
/// @brief convert int id to char id
/// @param id int id
/// @return char id
char id2name(int id) {
    if (id < 10) {
        return id + '0';
    } else {
        return id - 10 + 'A';
    }
}

/// @brief print board
/// @param board board
void print(const board_t board) {
    // 0: empty, 1/2: prev p1/p2 piece, 3/4: cur p1/p2 piece
    char* ch[5] = {
        " ",
        "\033[" CLI_COLOR_GREEN "mo\033[0m",
        "\033[" CLI_COLOR_RED "mx\033[0m",
        "\033[" CLI_STYLE_UNDERLINE "m\033[" CLI_STYLE_BOLD ";" CLI_COLOR_GREEN
        "mo\033[0m",
        "\033[" CLI_STYLE_UNDERLINE "m\033[" CLI_STYLE_BOLD ";" CLI_COLOR_RED
        "mx\033[0m",
    };
    int top = 0, bottom = BOARD_SIZE, left = 0, right = BOARD_SIZE;
    wrap_area(board, &top, &bottom, &left, &right, 3);
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
            printf(" %s |%c", ch[board[i][j]], "\n"[j != right - 1]);
        }
        for (int j = left; j < right + 1; j++) {
            printf("---+%c", "\n"[j != right]);
        }
    }
}

/// @brief eliminate the marks of whether a pos is just put by a player
/// @param board board before eliminate
void refresh(board_t board) {
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (board[i][j] > 2) board[i][j] -= 2;
        }
    }
}

/// @brief check if a pos is in board
/// @param pos
/// @return
bool inboard(point_t pos) {
    int n = BOARD_SIZE;
    return pos.x >= 0 && pos.x < n && pos.y >= 0 && pos.y < n;
}

bool available(board_t board, point_t pos) {
    if (!inboard(pos) || board[pos.x][pos.y]) return false;
    return true;
}

/// @brief put a piece on board
/// @param board board before putting
/// @param id put id
/// @param pos put position
/// @return
void put(board_t board, int id, point_t pos) {
    board[pos.x][pos.y] = id + 2;
}

/// @brief check if the board is draw (no available position)
/// @param board current board
/// @return 0 for not draw, 1 for draw
int check_draw(const board_t board) {
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (!board[i][j]) return 0;
        }
    }
    return 1;
}

/// @brief check whether the position is win for a player
/// @param board current board
/// @param pos position to check
/// @return 1 for player 1 win, -1 for player 2 win, 0 for no win
int check(const board_t board, point_t pos) {
    int id = board[pos.x][pos.y];
    if (!id) return 0;
    static const int arrows[4][2] = {{0, 1}, {1, 0}, {1, 1}, {-1, 1}};
    for (int i = 0, a, b, cnt; i < 4; i++) {
        a = arrows[i][0], b = arrows[i][1];
        point_t np = {pos.x, pos.y};
        for (cnt = 0; inboard(np); np.x += a, np.y += b) {
            if (board[np.x][np.y] == id)
                cnt++;
            else
                break;
        }
        np = (point_t){pos.x - a, pos.y - b};
        for (; inboard(np); np.x -= a, np.y -= b) {
            if (board[np.x][np.y] == id)
                cnt++;
            else
                break;
        }
        if (cnt >= WIN_LENGTH) return id == 1 ? 1 : -1;
    }
    return 0;
}

/// @brief check whether position is banned
/// @param board current board
/// @param pos position to check
/// @param id current id
/// @return 0 for accept, 1 for banned long, 2 for banned 33, 3 for banned 44
int banned(const board_t board, point_t pos, int id) {
    // print(board);
    // log("pos: (%d, %d), id: %d", pos.x, pos.y, id);
    if (id == -1) return POS_ACCEPT;
    board_t bd;
    memcpy(bd, board, sizeof(board_t));
    static const int arrows[4][2] = {{0, 1}, {1, 0}, {1, 1}, {-1, 1}};
    int cnt[8];
    for (int i = 0, a, b; i < 8; i++) {
        if (i < 4)
            a = arrows[i][0], b = arrows[i][1];
        else
            a = -arrows[i - 4][0], b = -arrows[i - 4][1];
        point_t np = {pos.x + a, pos.y + b};
        for (cnt[i] = 0; inboard(np); np.x += a, np.y += b) {
            if (bd[np.x][np.y] == id) {
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
    bd[pos.x][pos.y] = id;
    static const int live3s[3][10] = {
        {5, 0, 1, 1, 1, 0},
        {6, 0, 1, 0, 1, 1, 0},
        {6, 0, 1, 1, 0, 1, 0},
    };
    static const int exist4s[8][10] = {
        {5, 0, 1, 1, 1, 1},
        {5, 1, 0, 1, 1, 1},
        {5, 1, 1, 0, 1, 1},
        {5, 1, 1, 1, 0, 1},
        {5, 1, 1, 1, 1, 0},
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
                    if (bd[np.x][np.y] == (live3[k] ? id : 0)) {
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
                    if (bd[np.x][np.y] == (exist4[k] ? id : 0)) {
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
    bd[pos.x][pos.y] = 0;
    if (live3_cnt > 1) {
        return POS_BANNED_33;
    }
    if (exist4_cnt > 1) {
        return POS_BANNED_44;
    }
    return POS_ACCEPT;
}

/*
void test_ban(void) {
    int n = 5;
    struct {
        board_t board;
        point_t pos;
        int id;
    } tests[10] = {
        {
            {
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 1, 0, 0, 0},
                {0, 0, 0, 1, 0, 0, 0},
                {0, 0, 0, 1, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 1, 0, 0, 0},
                {0, 0, 0, 1, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
            },
            {5, 3},
            POS_BANNED_LONG,
        },
        {
            {
                {0, 0, 0, 0, 0, 0, 0},
                {1, 0, 0, 0, 0, 0, 0},
                {0, 1, 0, 0, 0, 0, 0},
                {0, 0, 1, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 1, 0, 0},
                {0, 0, 0, 0, 0, 1, 0},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
            },
            {4, 3},
            POS_BANNED_LONG,
        },
        {
            {
                {0, 0, 0, 0, 0, 0, 0},
                {1, 0, 0, 0, 0, 0, 0},
                {0, 1, 0, 0, 0, 0, 0},
                {0, 0, 1, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 1, 0},
                {0, 0, 0, 0, 0, 0, 1},
                {0, 0, 0, 0, 0, 0, 0, 1},
                {0, 0, 0, 0, 0, 0, 0},
            },
            {4, 3},
            POS_BANNED_44,
        },
        {{
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 1, 1, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 1, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 1, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
         },
         {4, 4},
         POS_BANNED_33},
        {{
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 2, 1, 1, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 1, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 1, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, 0, 0, 0, 0, 0, 0, 0},
         },
         {4, 4},
         POS_ACCEPT},
    };
    for (int i = 0; i < n; i++) {
        log("i = %d", i);
        int ban = banned(tests[i].board, tests[i].pos, 1);
        log("ban = %d (%s)", ban, POS_BAN_MSG[ban]);
        assert(ban == tests[i].id);
    }
}
*/
