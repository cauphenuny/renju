// author: Cauphenuny
// date: 2024/07/27

#include "board.h"

#include "util.h"

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#define printf(...) log_s(__VA_ARGS__)

/// @brief print {board} with one emphasized position {pos} or predicted probability
void print_implement(const board_t board, point_t emph_pos, const fboard_t prob)
{
    if (log_locked()) return;
#define dark(x) DARK x RESET
    const char* border_line[3][5] = {{dark("╔"), dark("═"), dark("╤"), dark("═"), dark("╗")},   //
                                     {dark("╟"), dark("─"), dark("┼"), dark("─"), dark("╢")},   //
                                     {dark("╚"), dark("═"), dark("╧"), dark("═"), dark("╝")}};  //
    const char* piece_ch[5] = {
        " ",                // empty
        GREEN "o" RESET,    // previous p1 pieces
        RED "x" RESET,      // previous p2 pieces
        L_GREEN "o" RESET,  // current p1 piece
        L_RED "x" RESET,    // current p2 piece
    };
#define LEVELS 7
    const struct {
        double thresh;
        const char* ch;
    } levels[LEVELS] = {
        {0.01, dark("#")},                  // 0.01 < p <= 0.05
        {0.05, "#"},                        // 0.05 < p <= 0.20
        {0.10, BOLD "#" RESET},             // 0.05 < p <= 0.20
        {0.20, L_YELLOW "#" RESET},         // 0.20 < p <= 0.50
        {0.50, L_RED "#" RESET},            // 0.50 < p <= 0.85
        {0.85, UNDERLINE L_RED "#" RESET},  // 0.85 < p <= 1
        {1.00, "?"},                        // invalid
    };
#undef dark
    for (int i = BOARD_SIZE - 1, line_type, col_type; i >= 0; i--) {
        switch (i) {
            case BOARD_SIZE - 1: line_type = 0; break;
            case 0: line_type = 2; break;
            default: line_type = 1;
        }
#if defined(DEBUG_LEVEL) && DEBUG_LEVEL >= 2
        printf("%2d%c", i, " ["[emph_pos.x == i && emph_pos.y == 0]);
#else
        printf("%2d%c", i + 1, " ["[emph_pos.x == i && emph_pos.y == 0]);
#endif
        for (int j = 0; j < 2 * BOARD_SIZE - 1; j++) {
            switch (j) {
                case 0: col_type = 0; break;
                case 2 * BOARD_SIZE - 2: col_type = 4; break;
                default: col_type = 2 + (j & 1);
            }
            if (j & 1) {
                if (emph_pos.x == i && emph_pos.y == j / 2)
                    printf("]");
                else if (emph_pos.x == i && emph_pos.y == (j + 1) / 2)
                    printf("[");
                else
                    printf("%s", border_line[line_type][col_type]);
            } else if (board[i][j / 2]) {
                printf("%s", piece_ch[board[i][j / 2]]);
            } else if (prob[i][j / 2] > levels[0].thresh) {
                int level = 0;
                for (int k = 0; k < LEVELS; k++) {
                    if (prob[i][j / 2] > levels[k].thresh) level = k;
                }
                printf("%s", levels[level].ch);
            } else {
                printf("%s", border_line[line_type][col_type]);
            }
        }
        printf("%c\n", " ]"[emph_pos.x == i && emph_pos.y == BOARD_SIZE - 1]);
    }
    printf("   ");
    for (int i = 0; i < 2 * BOARD_SIZE - 1; i++) {
#if defined(DEBUG_LEVEL) && DEBUG_LEVEL >= 2
        if (!(i & 1)) printf("%-2d", i / 2);
#else
        printf("%c", (i & 1) ? ' ' : 'A' + i / 2);
#endif
    }
    printf("\n");
#undef LEVELS
}

/// @brief print {board} with one emphasized position
void emphasis_print(const board_t board, point_t emph_pos)
{
    fboard_t prob;
    memset(prob, 0, sizeof(prob));
    print_implement(board, emph_pos, prob);
}

/// @brief print {board} predicted probability
void probability_print(const board_t board, const fboard_t prob)
{
    print_implement(board, (point_t){-1, -1}, prob);
}

/// @brief print {board} without emphasis
void print(const board_t board) { return emphasis_print(board, (point_t){-1, -1}); }

/// @brief get a minimal area [{begin}, {end}) that wraps up pieces in {board}
/// @param begin left-top corner of wrapped area
/// @param end right-bottom corner of wrapped area
/// @param margin min margin
void wrap_area(const board_t board, point_t* begin, point_t* end, int8_t margin)
{
    const int8_t n = BOARD_SIZE, mid = n / 2;
    begin->x = begin->y = mid - margin;
    end->x = end->y = mid + margin + 1;
    chkmin(end->y, n), chkmin(end->x, n);
    chkmax(begin->x, 0), chkmax(begin->y, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (board[i][j]) {
                chkmin(begin->x, max(0, i - margin));
                chkmin(begin->y, max(0, j - margin));
                chkmax(end->x, min(i + margin + 1, n));
                chkmax(end->y, min(j + margin + 1, n));
            }
        }
    }
}

/// @brief check if {pos} is in the board and is empty
bool available(const board_t board, point_t pos)
{
    if (!in_board(pos) || board[pos.x][pos.y]) return false;
    return true;
}

/// @brief put a player{id}'s piece at {pos} on {board}
void put(board_t board, int id, point_t pos) { board[pos.x][pos.y] = id; }

/// @brief check if {board} is a draw situation
bool is_draw(const board_t board)
{
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (!board[i][j]) return false;
        }
    }
    return true;
}

/// @brief check if {b1} is equal to {b2} at every position
bool is_equal(const board_t b1, const board_t b2)
{
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (b1[i][j] != b2[i][j]) {
                return false;
            }
        }
    }
    return true;
}

/// @brief check if {board} has available position for player{id}
bool have_space(const board_t board, int id)
{
    board_t dyn_board;
    memcpy(dyn_board, board, sizeof(board_t));
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (!board[i][j] && !is_forbidden(dyn_board, (point_t){i, j}, id, false)) {
                return true;
            }
        }
    }
    // print(board), prompt_pause();
    return false;
}

const char* pattern_typename[] = {
    [PAT_ETY] = "empty",   [PAT_44] = "double 4", [PAT_ATL] = "almost overline",
    [PAT_TL] = "overline", [PAT_D1] = "dead 1",   [PAT_A1] = "alive 1",
    [PAT_D2] = "dead 2",   [PAT_A2] = "alive 2",  [PAT_D3] = "dead 3",
    [PAT_A3] = "alive 3",  [PAT_D4] = "dead 4",   [PAT_A4] = "alive 4",
    [PAT_5] = "connect 5",
};

const char* pattern4_typename[] = {[PAT4_OTHERS] = "others",
                                   [PAT4_WIN] = "win",
                                   [PAT4_A33] = "double 3",
                                   [PAT4_44] = "double 4",
                                   [PAT4_TL] = "overline"};

static int powers[SEGMENT_LEN];
#define UP_COL_SIZE 3
static int upgrade_col[PATTERN_SIZE][UP_COL_SIZE];
static int count[PATTERN_SIZE];
static pattern_t pattern_memo[PATTERN_SIZE];
static pattern4_t pattern4_memo[PAT_TYPE_SIZE][PAT_TYPE_SIZE][PAT_TYPE_SIZE][PAT_TYPE_SIZE];
static int pattern_initialized;

/// @brief generate a int value from segment
int segment_encode(segment_t s)
{
    const piece_t* a = s.pieces;
    int result = 0;
    for (int i = 0; i < SEGMENT_LEN; i++) {
        // result += a[i] * powers[i];
        result += a[i] * (1 << (i * 2));
    }
    return result;
}

/// @brief decode the segment from int value
segment_t segment_decode(int v)
{
    segment_t result;
    for (int i = 0; i < SEGMENT_LEN; i++) {
        result.pieces[i] = (piece_t)(v % (int)PIECE_SIZE);
        v /= (int)PIECE_SIZE;
    }
    return result;
}

/// @brief print a segment and its data
void segment_print(segment_t s)
{
    const char ch[4] = {'.', 'o', 'x', '?'};
    printf("%5d [%c", segment_encode(s), ch[s.pieces[0]]);
    for (int i = 1; i < SEGMENT_LEN; i++) {
        printf(" %c", ch[s.pieces[i]]);
    }
    const int idx = segment_encode(s);
    // printf("]: level = %s, \tcols: [%d, %d, %d]\n", pattern_typename[pattern_mem[idx]],
    //        from_col[idx][0], from_col[idx][1], from_col[idx][2]);
    printf("]: level = %s\n", pattern_typename[pattern_memo[idx]]);
}

static int update(int prev, int pos, piece_t new_piece)
{
    return prev + new_piece * (1 << (pos * 2));
}

/// @brief convert int value of segment to pattern type
pattern_t to_pattern(int segment_value)
{
    assert(pattern_initialized);
    return pattern_memo[segment_value];
}

/// @brief convert 4 values of segments at 4 directions to pattern4 type
pattern4_t to_pattern4(int x, int y, int u, int v)
{
    assert(pattern_initialized);
    return pattern4_memo[x][y][u][v];
}

/// @brief check if there is a winner on {board} at {pos}
/// @param board current board
/// @param pos position to check
/// @return 1 for player 1 win, -1 for player 2 win, 0 for no winner
int check(const board_t board, point_t pos)
{
    const int id = board[pos.x][pos.y];
    if (!id) return 0;
    const int8_t arrows[4][2] = {{0, 1}, {1, 0}, {1, 1}, {-1, 1}};
    int8_t dx, dy;
    for (int i = 0, cnt; i < 4; i++) {
        dx = arrows[i][0], dy = arrows[i][1];
        point_t np = {pos.x, pos.y};
        for (cnt = 0; in_board(np); np.x += dx, np.y += dy) {
            if (board[np.x][np.y] == id)
                cnt++;
            else
                break;
        }
        np = (point_t){pos.x - dx, pos.y - dy};
        for (; in_board(np); np.x -= dx, np.y -= dy) {
            if (board[np.x][np.y] == id)
                cnt++;
            else
                break;
        }
        if (cnt >= WIN_LENGTH) return id == 1 ? 1 : -1;
    }
    return 0;
}

/// @brief check if {pos} is forbidden for player{id}
/// @param board current board
/// @param pos position to check
/// @param id current id
/// @param enable_log enable logs when detects a forbidden pos
/// @return 0 for accept, pat4 if forbidden
int is_forbidden(board_t board, point_t pos, int id, bool enable_log)
{
    assert(pattern_initialized);
    assert(in_board(pos) && !board[pos.x][pos.y]);
    // print(board);
    static const int8_t mid = WIN_LENGTH - 1, arrow[4][2] = {{1, 1}, {1, -1}, {1, 0}, {0, 1}};

    pattern_t idx[4] = {PAT_ETY, PAT_ETY, PAT_ETY, PAT_ETY};
    pattern4_t pat4;
    segment_t seg[4] = {0};

    // emphasis_print(board, pos);
    // pause();
    board[pos.x][pos.y] = id;
    for (int8_t i = 0, dx, dy; i < 4; i++) {
        dx = arrow[i][0], dy = arrow[i][1];
        for (int8_t j = -WIN_LENGTH + 1; j < WIN_LENGTH; j++) {
            const point_t np = (point_t){pos.x + dx * j, pos.y + dy * j};
            if (!in_board(np))
                seg[i].pieces[mid + j] = OPPO_PIECE;
            else if (!board[np.x][np.y]) {
                seg[i].pieces[mid + j] = EMPTY_PIECE;
            } else {
                seg[i].pieces[mid + j] = board[np.x][np.y] == id ? SELF_PIECE : OPPO_PIECE;
            }
        }
        int value = segment_encode(seg[i]);
        idx[i] = to_pattern(segment_encode(seg[i]));
        if (idx[i] >= PAT_A3 && idx[i] <= PAT_A4) {
            int cols[2];
            get_upgrade_columns(value, cols, 2);
            for (int j = 0; j < 2; j++) {
                if (cols[j] != -1) {
                    const point_t np =
                        (point_t){pos.x + dx * (cols[j] - mid), pos.y + dy * (cols[j] - mid)};
                    if (is_forbidden(board, np, id, false)) {
                        idx[i] = PAT_ETY;  // incorrect, but enough for checking forbid
                        break;
                    }
                }
            }
        }
        pat4 = to_pattern4(idx[0], idx[1], idx[2], idx[3]);
        if (pat4 > PAT4_WIN) break;
    }
    board[pos.x][pos.y] = 0;

    if (pat4 <= PAT4_WIN) return 0;
    if (enable_log) {
        log("forbidden pos, reason: %s", pattern4_typename[pat4]);
        log("detailed information:");
        emphasis_print(board, pos);
        for (int i = 0; i < 4; i++) {
            segment_print(seg[i]);
        }
    }
    return (int)pat4;
}

int is_forbidden_legacy(const board_t board, point_t pos, int id, bool enable_log)
{
    board_t dyn_board;
    memcpy(dyn_board, board, sizeof(board_t));
    return is_forbidden(dyn_board, pos, id, enable_log);
}

/// @brief calculate pattern type and store it
void pattern_init()
{
    powers[0] = 1;
    for (int i = 1; i < SEGMENT_LEN; i++) {
        powers[i] = powers[i - 1] * PIECE_SIZE;
    }
#define print_array(...)                      \
    for (int i = 0; i < PATTERN_SIZE; i++) {  \
        if (pattern_mem[i]) {                 \
            print_segment(segment_decode(i)); \
        }                                     \
    }

    /// terminate state: overline
    for (int idx = 0; idx < PATTERN_SIZE; idx++) {
        for (int cover_start = 0; cover_start < SEGMENT_LEN - WIN_LENGTH; cover_start++) {
            segment_t line = segment_decode(idx);
            for (int i = 0; i < WIN_LENGTH; i++) {
                line.pieces[cover_start + i] = SELF_PIECE;
            }
            for (int cover_end = cover_start + WIN_LENGTH; cover_end < SEGMENT_LEN; cover_end++) {
                line.pieces[cover_end] = SELF_PIECE;
                const int new_idx = segment_encode(line);
                pattern_memo[new_idx] = PAT_TL;
            }
        }
    }
    /// terminate state: win
    for (int idx = 0; idx < PATTERN_SIZE; idx++) {
        for (int cover_start = 0; cover_start <= SEGMENT_LEN - WIN_LENGTH; cover_start++) {
            segment_t line = segment_decode(idx);
            for (int i = 0; i < WIN_LENGTH; i++) {
                line.pieces[cover_start + i] = SELF_PIECE;
            }
            const int new_idx = segment_encode(line);
            if (!pattern_memo[new_idx]) pattern_memo[new_idx] = PAT_5;
        }
    }
    // print_array();
    // prompt_pause();

    /// in a reverse order,
    /// to calculate states that can be transferred to current state {idx} before visiting {idx}
    for (int idx = PATTERN_SIZE - 1, left, right; idx >= 0; idx--) {
        // log("round %d, idx %d", round, idx);
        if (pattern_memo[idx]) continue;  /// is terminate state PAT_TL/PAT_5
        const segment_t line = segment_decode(idx);
        pattern_t parent_pattern = PAT_ETY;  /// best parent pattern
        count[idx] = 0;
        memset(upgrade_col[idx], -1, sizeof(upgrade_col[idx]));
        // fprintf(stderr, "cur: "), print_segment(line);
        // prompt_pause();

        for (left = 0, right = 1; left < SEGMENT_LEN; left = right + 1) {
            while (left < SEGMENT_LEN && line.pieces[left] == OPPO_PIECE) (left)++;
            right = left + 1;
            while (right < SEGMENT_LEN && line.pieces[right] != OPPO_PIECE) (right)++;
            if (right - left >= WIN_LENGTH) break;
        }
        // log("left = %d, right = %d", left, right);
        // prompt_pause();

        if (left >= SEGMENT_LEN) continue;

        int win_pos[2] = {0}, pos_cnt = 0;

        for (int col = left; col < right; col++) {
            if (line.pieces[col] != EMPTY_PIECE) continue;
            const int new_idx = update(idx, col, SELF_PIECE);
            if (pattern_memo[new_idx] == PAT_5 || pattern_memo[new_idx] == PAT_TL) {
                if (pos_cnt < 2) win_pos[pos_cnt++] = col;
            }
            if (pattern_memo[new_idx] > parent_pattern) {
                parent_pattern = pattern_memo[new_idx], count[idx] = 0;
                memset(upgrade_col[idx], -1, sizeof(upgrade_col[idx]));
            }
            if (pattern_memo[new_idx] == parent_pattern) {
                if (count[idx] < 3) upgrade_col[idx][count[idx]] = col;
                count[idx]++;
                // log("write col %d", col);
            }
        }
        switch (parent_pattern) {
            case PAT_TL:
                if (right - left < 8)
                    pattern_memo[idx] = PAT_ATL;
                else
                    pattern_memo[idx] = PAT_44;
                break;
            case PAT_5:
                if (count[idx] == 1)
                    pattern_memo[idx] = PAT_D4;
                else {
                    if (win_pos[1] - win_pos[0] >= WIN_LENGTH) {
                        pattern_memo[idx] = PAT_A4;
                    } else {
                        pattern_memo[idx] = PAT_44;
                    }
                }
                break;
            case PAT_A4: pattern_memo[idx] = PAT_A3; break;
            case PAT_D4: pattern_memo[idx] = PAT_D3; break;
            case PAT_A3: pattern_memo[idx] = PAT_A2; break;
            case PAT_D3: pattern_memo[idx] = PAT_D2; break;
            case PAT_A2: pattern_memo[idx] = PAT_A1; break;
            case PAT_D2: pattern_memo[idx] = PAT_D1; break;
            default: break;
        }
    }
    // print_array();
    // prompt_pause();
    for (int i = 0; i < PAT_TYPE_SIZE; i++) {
        for (int j = 0; j < PAT_TYPE_SIZE; j++) {
            for (int k = 0; k < PAT_TYPE_SIZE; k++) {
                for (int u = 0; u < PAT_TYPE_SIZE; u++) {
                    int cnt[PAT_TYPE_SIZE] = {0};
                    cnt[i]++, cnt[j]++, cnt[k]++, cnt[u]++;
                    if (cnt[PAT_5])
                        pattern4_memo[i][j][k][u] = PAT4_WIN;
                    else if (cnt[PAT_TL])
                        pattern4_memo[i][j][k][u] = PAT4_TL;
                    else if (cnt[PAT_A3] > 1)
                        pattern4_memo[i][j][k][u] = PAT4_A33;
                    else if ((cnt[PAT_A4] + cnt[PAT_D4]) > 1 || cnt[PAT_44])
                        pattern4_memo[i][j][k][u] = PAT4_44;
                    else
                        pattern4_memo[i][j][k][u] = PAT4_OTHERS;
                }
            }
        }
    }
    pattern_initialized = 1;
#undef print_array
    // test_forbid();
}

/// @brief get columns on which put a piece can upgrade pattern
/// @param segment_value the int value of segment
/// @param cols array that stores columns
/// @param limit array size
void get_upgrade_columns(int segment_value, int* cols, int limit)
{
    assert(pattern_initialized);
    memset(cols, -1, limit * sizeof(int));
    for (int i = 0, cur = 0; i < 3 && cur < limit; i++) {
        if (upgrade_col[segment_value][i] != -1) {
            cols[cur++] = upgrade_col[segment_value][i];
        }
    }
};

void get_patterns(const board_t board, point_t pos, pattern_t arr[])
{
    int id;
    if (!in_board(pos) || !((id = board[pos.x][pos.y]))) {
        arr[0] = arr[1] = arr[2] = arr[3] = PAT_ETY;
    } else {
        const int8_t arrows[4][2] = {{1, 1}, {1, -1}, {1, 0}, {0, 1}};
        int piece;
        for (int8_t i = 0, dx, dy; i < 4; i++) {
            dx = arrows[i][0], dy = arrows[i][1];
            int val = 0;
            for (int8_t j = -WIN_LENGTH + 1; j < WIN_LENGTH; j++) {
                const point_t np = (point_t){pos.x + dx * j, pos.y + dy * j};
                if (!in_board(np))
                    val = val * PIECE_SIZE + OPPO_PIECE;
                else if (!((piece = board[np.x][np.y])))
                    val = val * PIECE_SIZE + EMPTY_PIECE;
                else
                    val = val * PIECE_SIZE + ((piece == id) ? SELF_PIECE : OPPO_PIECE);
            }
            arr[i] = to_pattern(val);
        }
    }
}

/// @brief encode from raw board to compressed board
void encode(const board_t src, comp_board_t dest)
{
    for (int i = 0; i < BOARD_SIZE; i++) {
        dest[i] = 0;
        for (int j = BOARD_SIZE - 1; j >= 0; j--) {
            dest[i] = dest[i] * 4 + src[i][j];
        }
    }
}

/// @brief decode from compressed board to raw board
void decode(const comp_board_t src, board_t dest)
{
    for (int i = 0; i < BOARD_SIZE; i++) {
        line_t tmp = src[i];
        for (int j = 0; j < BOARD_SIZE; j++) {
            dest[i][j] = tmp & 3;
            tmp >>= 2;
        }
    }
}

void print_compressed_board(const comp_board_t board, point_t emph_pos)
{
    board_t b;
    decode(board, b);
    emphasis_print(b, emph_pos);
}
