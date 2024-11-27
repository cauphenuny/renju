#include "pattern.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>

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