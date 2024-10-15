// author: Cauphenuny
// date: 2024/07/27
#include "board.h"

#include "util.h"

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

/// @brief calculate the wrap area of chessboard: $[top, bottom) \times [left,
/// right)$
/// @param board chessboard
/// @param top return top
/// @param bottom return bottom
/// @param left return left
/// @param right return right
/// @param radius wrap radius
void wrap_area(const board_t board, point_t* begin, point_t* end, int8_t radius)
{
    int8_t n = BOARD_SIZE, mid = n / 2;
    begin->x = begin->y = mid - radius;
    end->x = end->y = mid + radius + 1;
    chkmin(end->y, n), chkmin(end->x, n);
    chkmax(begin->x, 0), chkmax(begin->y, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (board[i][j]) {
                chkmin(begin->x, max(0, i - radius));
                chkmin(begin->y, max(0, j - radius));
                chkmax(end->x, min(i + radius + 1, n));
                chkmax(end->y, min(j + radius + 1, n));
            }
        }
    }
}

/// @brief print board
/// @param board board
void emph_print(const board_t board, point_t pos)
{
#define dark(x) DARK x NONE
    static const char* border_line[3][5] = {
        {dark("╔"), dark("═"), dark("╤"), dark("═"), dark("╗")},   //
        {dark("╟"), dark("─"), dark("┼"), dark("─"), dark("╢")},   //
        {dark("╚"), dark("═"), dark("╧"), dark("═"), dark("╝")}};  //
#undef dark
    // 0: empty, 1/2: prev p1/p2 piece, 3/4: cur p1/p2 piece
    static const char* ch[5] = {
        " ",               //
        GREEN "o" NONE,    //
        RED "x" NONE,      //
        L_GREEN "o" NONE,  //
        L_RED "x" NONE,    //
    };
    // point_t begin = {0, 0}, end = {BOARD_SIZE, BOARD_SIZE};
    // wrap_area(board, &begin, &end, 3);
    // int8_t left = begin.y, right = end.y, top = begin.x, bottom = end.x;
    for (int i = BOARD_SIZE - 1, line_type, col_type; i >= 0; i--) {
        switch (i) {
            case BOARD_SIZE - 1: line_type = 0; break;
            case 0: line_type = 2; break;
            default: line_type = 1;
        }
#ifdef DEBUG
        printf("%2d%c", i, " ["[pos.x == i && pos.y == 0]);
#else
        printf("%2d%c", i + 1, " ["[pos.x == i && pos.y == 0]);
#endif
        for (int j = 0; j < 2 * BOARD_SIZE - 1; j++) {
            switch (j) {
                case 0: col_type = 0; break;
                case 2 * BOARD_SIZE - 2: col_type = 4; break;
                default: col_type = 2 + (j & 1);
            }
            if ((j & 1) && pos.x == i && pos.y == j / 2)
                printf("]");
            else if ((j & 1) && pos.x == i && pos.y == (j + 1) / 2)
                printf("[");
            else
                printf("%s", ((j & 1) || !board[i][j / 2]) ? border_line[line_type][col_type]
                                                           : ch[board[i][j / 2]]);
        }
        printf("%c\n", " ]"[pos.x == i && pos.y == BOARD_SIZE - 1]);
    }
    printf("   ");
    for (int i = 0; i < 2 * BOARD_SIZE - 1; i++) {
#ifdef DEBUG
        if (!(i & 1)) printf("%-2d", i / 2);
#else
        printf("%c", (i & 1) ? ' ' : 'A' + i / 2);
#endif
    }
    printf("\n");
}

void print(const board_t board)
{
    return emph_print(board, (point_t){-1, -1});
}

/// @brief check if a pos is in board
/// @param pos
/// @return
bool inboard(point_t pos)
{
    int n = BOARD_SIZE;
    return pos.x >= 0 && pos.x < n && pos.y >= 0 && pos.y < n;
}

bool available(board_t board, point_t pos)
{
    if (!inboard(pos) || board[pos.x][pos.y]) return false;
    return true;
}

/// @brief put a piece on board
/// @param board board before putting
/// @param id put id
/// @param pos put position
void put(board_t board, int id, point_t pos)
{
    board[pos.x][pos.y] = id;
}

/// @brief check if the board is draw (no available position)
/// @param board current board
/// @return 0 for not draw, 1 for draw
int check_draw(const board_t board)
{
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (!board[i][j]) return 0;
        }
    }
    return 1;
}

const char* pattern_typename[] = {
    [PAT_ETY] = "empty",   [PAT_44] = "double 4", [PAT_ATL] = "almost overline",
    [PAT_TL] = "overline", [PAT_D1] = "dead 1",   [PAT_A1] = "alive 1",
    [PAT_D2] = "dead 2",   [PAT_A2] = "alive 2",  [PAT_D3] = "dead 3",
    [PAT_A3] = "alive 3",  [PAT_D4] = "dead 4",   [PAT_A4] = "alive 4",
    [PAT_5] = "connect 5",
};

const char* pattern4_typename[] = {
    [PAT4_OTHERS] = "others", [PAT4_43] = "4 and 3",  [PAT4_WIN] = "win",
    [PAT4_A33] = "double 3",  [PAT4_44] = "double 4", [PAT4_TL] = "overline"};

static int power3[PATTERN_LEN];
#define FROM_COL_SIZE 3
static int from_col[PATTERN_SIZE][FROM_COL_SIZE];
static int count[PATTERN_SIZE];
static int pattern_mem[PATTERN_SIZE];
static int pattern4_mem[PAT_SIZE][PAT_SIZE][PAT_SIZE][PAT_SIZE];
static int pattern_initialized;

int segment_encode(segment_t s)
{
    int* a = s.data;
    int result = 0;
    for (int i = 0; i < PATTERN_LEN; i++) {
        result = result * 3 + a[i];
    }
    return result;
}

segment_t segment_decode(int v)
{
    segment_t result;
    for (int i = PATTERN_LEN - 1; i >= 0; i--) {
        result.data[i] = v % 3;
        v /= 3;
    }
    return result;
}

static void print_segment(segment_t s)
{
    char ch[3] = {'.', 'o', 'x'};
    printf("%5d [%c", segment_encode(s), ch[s.data[0]]);
    for (int i = 1; i < PATTERN_LEN; i++) {
        printf(" %c", ch[s.data[i]]);
    }
    int idx = segment_encode(s);
    // printf("]: level = %s, \tcols: [%d, %d, %d]\n", pattern_typename[pattern_mem[idx]],
    //        from_col[idx][0], from_col[idx][1], from_col[idx][2]);
    printf("]: level = %s\n", pattern_typename[pattern_mem[idx]]);
}

static int update(int prev, int pos, int new_piece)
{
    return prev + new_piece * power3[PATTERN_LEN - 1 - pos];
}

int to_pattern(int x)
{
    assert(pattern_initialized);
    return pattern_mem[x];
}

int to_pattern4(int x, int y, int u, int v)
{
    assert(pattern_initialized);
    return pattern4_mem[x][y][u][v];
}

/// @brief check whether the position is win for a player
/// @param board current board
/// @param pos position to check
/// @return 1 for player 1 win, -1 for player 2 win, 0 for no win
int check(const board_t board, point_t pos)
{
    int id = board[pos.x][pos.y];
    if (!id) return 0;
    static const int8_t arrows[4][2] = {{0, 1}, {1, 0}, {1, 1}, {-1, 1}};
    int8_t dx, dy;
    for (int i = 0, cnt; i < 4; i++) {
        dx = arrows[i][0], dy = arrows[i][1];
        point_t np = {pos.x, pos.y};
        for (cnt = 0; inboard(np); np.x += dx, np.y += dy) {
            if (board[np.x][np.y] == id)
                cnt++;
            else
                break;
        }
        np = (point_t){pos.x - dx, pos.y - dy};
        for (; inboard(np); np.x -= dx, np.y -= dy) {
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
int is_banned(const board_t board, point_t pos, int id, bool enable_log)
{
    assert(pattern_initialized);
    // print(board);
    static int8_t arrow[4][2] = {{1, 1}, {1, -1}, {1, 0}, {0, 1}};
    int idx[4];
    int mid = WIN_LENGTH - 1;
    static segment_t seg[4];
    for (int8_t i = 0, a, b; i < 4; i++) {
        a = arrow[i][0], b = arrow[i][1];
        for (int8_t j = -WIN_LENGTH + 1; j < WIN_LENGTH; j++) {
            point_t np = (point_t){pos.x + a * j, pos.y + b * j};
            if (!inboard(np))
                seg[i].data[mid + j] = OPPO_POS;
            else if (!board[np.x][np.y])
                seg[i].data[mid + j] = EMPTY_POS;
            else
                seg[i].data[mid + j] = board[np.x][np.y] == id ? SELF_POS : OPPO_POS;
        }
        seg[i].data[mid] = SELF_POS;
        // print_segment(seg);
        idx[i] = to_pattern(segment_encode(seg[i]));
    }
    int pat4 = to_pattern4(idx[0], idx[1], idx[2], idx[3]);
    if (pat4 <= PAT4_WIN) return 0;
    if (enable_log) {
        log("forbidden pos, reason: %s", pattern4_typename[pat4]);
        for (int i = 0; i < 4; i++) {
            print_segment(seg[i]);
        }
        printf("%s | %s | %s | %s\n", pattern_typename[idx[0]], pattern_typename[idx[1]],
            pattern_typename[idx[2]], pattern_typename[idx[3]]);
    }
    return pat4;
}

void pattern_init()
{
    power3[0] = 1;
    for (int i = 1; i < PATTERN_LEN; i++) {
        power3[i] = power3[i - 1] * 3;
    }
#define print_array(...)                      \
    for (int i = 0; i < PATTERN_SIZE; i++) {  \
        if (pattern_mem[i]) {                 \
            print_segment(segment_decode(i)); \
        }                                     \
    }

    for (int idx = 0; idx < PATTERN_SIZE; idx++) {
        for (int cover_start = 0; cover_start < PATTERN_LEN - WIN_LENGTH; cover_start++) {
            segment_t line = segment_decode(idx);
            for (int i = 0; i < WIN_LENGTH; i++) {
                line.data[cover_start + i] = SELF_POS;
            }
            for (int cover_end = cover_start + WIN_LENGTH; cover_end < PATTERN_LEN; cover_end++) {
                line.data[cover_end] = SELF_POS;
                int new_idx = segment_encode(line);
                pattern_mem[new_idx] = PAT_TL;
            }
        }
    }
    for (int idx = 0; idx < PATTERN_SIZE; idx++) {
        for (int cover_start = 0; cover_start <= PATTERN_LEN - WIN_LENGTH; cover_start++) {
            segment_t line = segment_decode(idx);
            for (int i = 0; i < WIN_LENGTH; i++) {
                line.data[cover_start + i] = SELF_POS;
            }
            int new_idx = segment_encode(line);
            if (!pattern_mem[new_idx]) pattern_mem[new_idx] = PAT_5;
        }
    }
    // print_array();
    // prompt_pause();

    for (int idx = PATTERN_SIZE - 1, left, right; idx >= 0; idx--) {
        // log("round %d, idx %d", round, idx);
        if (pattern_mem[idx]) continue;
        segment_t line = segment_decode(idx);
        int parent_pattern = 0;
        count[idx] = 0;
        memset(from_col[idx], -1, sizeof(from_col[idx]));
        // fprintf(stderr, "cur: "), print_segment(line);
        // prompt_pause();

        for (left = 0, right = 1; left < PATTERN_LEN; left = right + 1) {
            while (left < PATTERN_LEN && line.data[left] == OPPO_POS) (left)++;
            right = left + 1;
            while (right < PATTERN_LEN && line.data[right] != OPPO_POS) (right)++;
            if (right - left >= WIN_LENGTH) break;
        }
        // log("left = %d, right = %d", left, right);
        // prompt_pause();

        if (left >= PATTERN_LEN) continue;

        int win_pos[2] = {0}, pos_cnt = 0;

        for (int col = left; col < right; col++) {
            if (line.data[col] != EMPTY_POS) continue;
            int new_idx = update(idx, col, SELF_POS);
            if (pattern_mem[new_idx] == PAT_5 || pattern_mem[new_idx] == PAT_TL) {
                if (pos_cnt < 2) win_pos[pos_cnt++] = col;
            }
            if (pattern_mem[new_idx] > parent_pattern) {
                parent_pattern = pattern_mem[new_idx], count[idx] = 0;
                memset(from_col[idx], -1, sizeof(from_col[idx]));
            }
            if (pattern_mem[new_idx] == parent_pattern) {
                if (count[idx] < 3) from_col[idx][count[idx]] = col;
                count[idx]++;
                // log("write col %d", col);
            }
        }
        switch (parent_pattern) {
            case PAT_TL:
                if (right - left < 8)
                    pattern_mem[idx] = PAT_ATL;
                else
                    pattern_mem[idx] = PAT_44;
                break;
            case PAT_5:
                if (count[idx] == 1)
                    pattern_mem[idx] = PAT_D4;
                else {
                    if (win_pos[1] - win_pos[0] >= WIN_LENGTH) {
                        pattern_mem[idx] = PAT_A4;
                    } else {
                        pattern_mem[idx] = PAT_44;
                    }
                }
                break;
            case PAT_A4: pattern_mem[idx] = PAT_A3; break;
            case PAT_D4: pattern_mem[idx] = PAT_D3; break;
            case PAT_A3: pattern_mem[idx] = PAT_A2; break;
            case PAT_D3: pattern_mem[idx] = PAT_D2; break;
            case PAT_A2: pattern_mem[idx] = PAT_A1; break;
            case PAT_D2: pattern_mem[idx] = PAT_D1; break;
            default: break;
        }
    }
    // print_array();
    // prompt_pause();
    for (int i = 0; i < PAT_SIZE; i++) {
        for (int j = 0; j < PAT_SIZE; j++) {
            for (int k = 0; k < PAT_SIZE; k++) {
                for (int u = 0; u < PAT_SIZE; u++) {
                    int cnt[PAT_SIZE] = {0};
                    cnt[i]++, cnt[j]++, cnt[k]++, cnt[u]++;
                    if (cnt[PAT_5])
                        pattern4_mem[i][j][k][u] = PAT4_WIN;
                    else if (cnt[PAT_TL])
                        pattern4_mem[i][j][k][u] = PAT4_TL;
                    else if (cnt[PAT_A3] > 1)
                        pattern4_mem[i][j][k][u] = PAT4_A33;
                    else if ((cnt[PAT_A4] + cnt[PAT_D4]) > 1 || cnt[PAT_44])
                        pattern4_mem[i][j][k][u] = PAT4_44;
                    else if ((cnt[PAT_A4] + cnt[PAT_D4]) && (cnt[PAT_A3]))
                        pattern4_mem[i][j][k][u] = PAT4_43;
                    else
                        pattern4_mem[i][j][k][u] = PAT4_OTHERS;
                }
            }
        }
    }
    pattern_initialized = 1;
#undef print_array
    // test_ban();
}

void get_critical_column(int pattern, int* cols, int limit)
{
    assert(pattern_initialized);
    memset(cols, -1, limit * sizeof(int));
    for (int i = 0, cur = 0; i < 3 && cur < limit; i++) {
        if (from_col[pattern][i] != -1) {
            cols[cur++] = from_col[pattern][i];
        }
    }
};