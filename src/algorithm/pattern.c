#include "pattern.h"

#include "board.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char* pattern_typename[] = {
    [PAT_EMPTY] = "empty", [PAT_44] = "double 4", [PAT_ATL] = "almost overline",
    [PAT_TL] = "overline", [PAT_D1] = "dead 1",   [PAT_A1] = "alive 1",
    [PAT_D2] = "dead 2",   [PAT_A2] = "alive 2",  [PAT_D3] = "dead 3",
    [PAT_A3] = "alive 3",  [PAT_D4] = "dead 4",   [PAT_A4] = "alive 4",
    [PAT_WIN] = "win 5",
};

const char* pattern4_typename[] = {[PAT4_OTHERS] = "others",
                                   [PAT4_WIN] = "win",
                                   [PAT4_A33] = "double 3",
                                   [PAT4_44] = "double 4",
                                   [PAT4_TL] = "overline"};

#define COL_STORAGE_SIZE SEGMENT_LEN

#define HALF (WIN_LENGTH - 1)

typedef struct {
    int upgrade_col[SEGMENT_MASK][COL_STORAGE_SIZE];
    int upgrade_col_cnt[SEGMENT_MASK];
    int consist_col[SEGMENT_MASK][COL_STORAGE_SIZE];
    int consist_col_cnt[SEGMENT_MASK];
    int defend_col[SEGMENT_MASK][COL_STORAGE_SIZE];
    int defend_col_cnt[SEGMENT_MASK];
    pattern_t pattern[SEGMENT_MASK];
    pattern_t parent_pattern[SEGMENT_MASK];
    pattern4_t pattern4[PAT_TYPE_SIZE][PAT_TYPE_SIZE][PAT_TYPE_SIZE][PAT_TYPE_SIZE];
} memo_t;

memo_t forbid, no_forbid;
static int pattern_initialized;

static int update(int prev, int pos, piece_t new_piece) {
    return prev + new_piece * (1 << (pos * 2));
}

static void dp(memo_t* memo, bool consider_forbid) {
    /// in a reverse order,
    /// to calculate states that can be transferred to current state {idx} before visiting {idx}
    for (int idx = SEGMENT_MASK - 1, left, right; idx >= 0; idx--) {
        // log("round %d, idx %d", round, idx);
        const segment_t line = decode_segment(idx);
        for (left = 0, right = 1; left < SEGMENT_LEN; left = right + 1) {
            while (left < SEGMENT_LEN && line.pieces[left] == OPPO_PIECE) (left)++;
            right = left + 1;
            while (right < SEGMENT_LEN && line.pieces[right] != OPPO_PIECE) (right)++;
            if (right - left >= WIN_LENGTH) break;
        }
        if (memo->pattern[idx]) {
            for (int col = left; col < right; col++) {
                if (line.pieces[col] == SELF_PIECE) {
                    memo->consist_col[idx][memo->consist_col_cnt[idx]++] = col;
                }
            }
            continue;  /// is terminate state PAT_TL/PAT_WIN
        }

        pattern_t parent_pattern = PAT_EMPTY;  /// best parent pattern
        memo->upgrade_col_cnt[idx] = 0;
        memset(memo->upgrade_col[idx], -1, sizeof(memo->upgrade_col[idx]));
        // fprintf(stderr, "cur: "), print_segment(line);
        // prompt_pause();

        // log("left = %d, right = %d", left, right);
        // prompt_pause();

        if (left >= SEGMENT_LEN) continue;

        int win_pos[2] = {0}, pos_cnt = 0;

        for (int col = left; col < right; col++) {
            if (line.pieces[col] != EMPTY_PIECE) continue;
            const int new_idx = update(idx, col, SELF_PIECE);
            if (memo->pattern[new_idx] == PAT_WIN || memo->pattern[new_idx] == PAT_TL) {
                if (pos_cnt < 2) win_pos[pos_cnt++] = col;
            }
            if (memo->pattern[new_idx] > parent_pattern) {
                parent_pattern = memo->pattern[new_idx], memo->upgrade_col_cnt[idx] = 0;
                memset(memo->upgrade_col[idx], -1, sizeof(memo->upgrade_col[idx]));
            }
            if (memo->pattern[new_idx] == parent_pattern) {
                if (memo->upgrade_col_cnt[idx] < COL_STORAGE_SIZE)
                    memo->upgrade_col[idx][memo->upgrade_col_cnt[idx]] = col,
                    memo->upgrade_col_cnt[idx]++;
                // log("write col %d", col);
            }
        }

        memo->parent_pattern[idx] = parent_pattern;
        switch (parent_pattern) {
            case PAT_TL:
                if (right - left < 8)
                    memo->pattern[idx] = PAT_ATL;
                else
                    memo->pattern[idx] = PAT_44;
                break;
            case PAT_WIN:
                if (memo->upgrade_col_cnt[idx] == 1)
                    memo->pattern[idx] = PAT_D4;
                else {
                    if (!consider_forbid || win_pos[1] - win_pos[0] >= WIN_LENGTH) {
                        memo->pattern[idx] = PAT_A4;
                    } else {
                        memo->pattern[idx] = PAT_44;
                    }
                }
                break;
            case PAT_A4: memo->pattern[idx] = PAT_A3; break;
            case PAT_D4: memo->pattern[idx] = PAT_D3; break;
            case PAT_A3: memo->pattern[idx] = PAT_A2; break;
            case PAT_D3: memo->pattern[idx] = PAT_D2; break;
            case PAT_A2: memo->pattern[idx] = PAT_A1; break;
            case PAT_D2: memo->pattern[idx] = PAT_D1; break;
            default: break;
        }

        for (int col = left; col < right; col++) {
            if (line.pieces[col] == SELF_PIECE) {
                memo->consist_col[idx][memo->consist_col_cnt[idx]++] = col;
            }
            if (line.pieces[col] == EMPTY_PIECE) {
                int new_idx = update(idx, col, OPPO_PIECE);
                if (memo->pattern[idx] != memo->pattern[new_idx]) {
                    memo->defend_col[idx][memo->defend_col_cnt[idx]++] = col;
                }
            }
        }
    }

    for (int i = 0; i < PAT_TYPE_SIZE; i++) {
        for (int j = 0; j < PAT_TYPE_SIZE; j++) {
            for (int k = 0; k < PAT_TYPE_SIZE; k++) {
                for (int u = 0; u < PAT_TYPE_SIZE; u++) {
                    int cnt[PAT_TYPE_SIZE] = {0};
                    cnt[i]++, cnt[j]++, cnt[k]++, cnt[u]++;
                    if (consider_forbid) {
                        if (cnt[PAT_WIN])
                            memo->pattern4[i][j][k][u] = PAT4_WIN;
                        else if (cnt[PAT_TL])
                            memo->pattern4[i][j][k][u] = PAT4_TL;
                        else if (cnt[PAT_A3] > 1)
                            memo->pattern4[i][j][k][u] = PAT4_A33;
                        else if ((cnt[PAT_A4] + cnt[PAT_D4]) > 1 || cnt[PAT_44])
                            memo->pattern4[i][j][k][u] = PAT4_44;
                        else
                            memo->pattern4[i][j][k][u] = PAT4_OTHERS;
                    } else {
                        if (cnt[PAT_WIN])
                            memo->pattern4[i][j][k][u] = PAT4_WIN;
                        else
                            memo->pattern4[i][j][k][u] = PAT4_OTHERS;
                    }
                }
            }
        }
    }
}

/// @brief calculate pattern type and store it
void pattern_init() {
    /// terminate state: overline > 5
    for (int idx = 0; idx < SEGMENT_MASK; idx++) {
        for (int cover_start = 0; cover_start < SEGMENT_LEN - WIN_LENGTH; cover_start++) {
            segment_t line = decode_segment(idx);
            for (int i = 0; i < WIN_LENGTH; i++) {
                line.pieces[cover_start + i] = SELF_PIECE;
            }
            for (int cover_end = cover_start + WIN_LENGTH; cover_end < SEGMENT_LEN; cover_end++) {
                line.pieces[cover_end] = SELF_PIECE;
                const int new_idx = encode_segment(line);
                forbid.pattern[new_idx] = PAT_TL;
            }
        }
    }
    /// terminate state: win == 5
    for (int idx = 0; idx < SEGMENT_MASK; idx++) {
        for (int cover_start = 0; cover_start <= SEGMENT_LEN - WIN_LENGTH; cover_start++) {
            segment_t line = decode_segment(idx);
            for (int i = 0; i < WIN_LENGTH; i++) {
                line.pieces[cover_start + i] = SELF_PIECE;
            }
            const int new_idx = encode_segment(line);
            if (!forbid.pattern[new_idx]) forbid.pattern[new_idx] = PAT_WIN;
        }
    }

    /// terminate state: win >=5
    for (int idx = 0; idx < SEGMENT_MASK; idx++) {
        for (int cover_start = 0; cover_start <= SEGMENT_LEN - WIN_LENGTH; cover_start++) {
            segment_t line = decode_segment(idx);
            for (int i = 0; i < WIN_LENGTH; i++) {
                line.pieces[cover_start + i] = SELF_PIECE;
            }
            for (int cover_end = cover_start + WIN_LENGTH - 1; cover_end < SEGMENT_LEN;
                 cover_end++) {
                line.pieces[cover_end] = SELF_PIECE;
                const int new_idx = encode_segment(line);
                no_forbid.pattern[new_idx] = PAT_WIN;
            }
        }
    }
    // prompt_pause();
    dp(&no_forbid, false);
    #ifdef NO_FORBID
    memcpy(&forbid, &no_forbid, sizeof(memo_t));
    #else
    dp(&forbid, true);
    #endif
    pattern_initialized = 1;
}

/// @brief generate a int value from segment
int encode_segment(segment_t s) {
    const piece_t* a = s.pieces;
    int result = 0;
    for (int i = 0; i < SEGMENT_LEN; i++) {
        // result += a[i] * powers[i];
        result += a[i] * (1 << (i * 2));
    }
    return result;
}

/// @brief decode the segment from int value
segment_t decode_segment(int v) {
    segment_t result;
    for (int i = 0; i < SEGMENT_LEN; i++) {
        result.pieces[i] = (piece_t)(v % (int)PIECE_SIZE);
        v /= (int)PIECE_SIZE;
    }
    return result;
}

bool segment_valid(segment_t s) {
    for (int i = 0; i < SEGMENT_LEN; i++) {
        if (s.pieces[i] != EMPTY_PIECE && s.pieces[i] != SELF_PIECE && s.pieces[i] != OPPO_PIECE) {
            return false;
        }
    }
    return true;
}

/// @brief print a segment and its data
void print_segment(segment_t s, bool consider_forbid) {
    memo_t* memo = consider_forbid ? &forbid : &no_forbid;
    const char ch[4] = {'-', 'o', 'x', '?'};
    printf("%5d [%c", encode_segment(s), ch[s.pieces[0]]);
    for (int i = 1; i < SEGMENT_LEN; i++) {
        printf(" %c", ch[s.pieces[i]]);
    }
    const int idx = encode_segment(s);
    // printf("]: level = %s, \tcols: [%d, %d, %d]\n", pattern_typename[pattern_mem[idx]],
    //        from_col[idx][0], from_col[idx][1], from_col[idx][2]);
    printf("]: level = %s, ", pattern_typename[memo->pattern[idx]]);
    printf("upd: [");
    for (int i = 0; i < memo->upgrade_col_cnt[idx]; i++) {
        printf("%d%s", memo->upgrade_col[idx][i], i == memo->upgrade_col_cnt[idx] - 1 ? "" : ", ");
    }
    printf("], con: [");
    for (int i = 0; i < memo->consist_col_cnt[idx]; i++) {
        printf("%d%s", memo->consist_col[idx][i], i == memo->consist_col_cnt[idx] - 1 ? "" : ", ");
    }
    printf("], def: [");
    for (int i = 0; i < memo->defend_col_cnt[idx]; i++) {
        printf("%d%s", memo->defend_col[idx][i], i == memo->defend_col_cnt[idx] - 1 ? "" : ", ");
    }
    printf("]\n");
}

segment_t get_segment(board_t board, point_t pos, int dx, int dy) {
    segment_t seg;
    const int id = board[pos.x][pos.y];
    for (int8_t j = -HALF; j <= HALF; j++) {
        const point_t np = (point_t){pos.x + dx * j, pos.y + dy * j};
        if (!in_board(np))
            seg.pieces[HALF + j] = OPPO_PIECE;
        else if (!board[np.x][np.y]) {
            seg.pieces[HALF + j] = EMPTY_PIECE;
        } else {
            seg.pieces[HALF + j] = board[np.x][np.y] == id ? SELF_PIECE : OPPO_PIECE;
        }
    }
    return seg;
}

/// @brief convert int value of segment to pattern type
pattern_t to_pattern(int segment_value, bool consider_forbid) {
    assert(pattern_initialized);
    if (consider_forbid)
        return forbid.pattern[segment_value];
    else
        return no_forbid.pattern[segment_value];
}

pattern_t to_upgraded_pattern(int segment_value, bool consider_forbid) {
    assert(pattern_initialized);
    if (consider_forbid)
        return forbid.parent_pattern[segment_value];
    else
        return no_forbid.parent_pattern[segment_value];
}

/// @brief convert 4 values of segments at 4 directions to pattern4 type
pattern4_t to_pattern4(int x, int y, int u, int v, bool consider_forbid) {
    assert(pattern_initialized);
    if (consider_forbid) {
        return forbid.pattern4[x][y][u][v];
    } else {
        return no_forbid.pattern4[x][y][u][v];
    }
}

/// @brief get columns on which put a piece can upgrade pattern
/// @param segment_value the int value of segment
/// @param cols array that stores columns
/// @param limit array size
void get_upgrade_columns(int segment_value, bool consider_forbid, int* cols, int limit) {
    assert(pattern_initialized);
    memset(cols, -1, limit * sizeof(int));
    memo_t* memo = consider_forbid ? &forbid : &no_forbid;
    for (int i = 0, cur = 0; i < COL_STORAGE_SIZE && cur < limit; i++) {
        int col = memo->upgrade_col[segment_value][i];
        if (col != -1) {
            cols[cur++] = col;
        }
    }
};

point_t column_to_point(point_t pos, int dx, int dy, int col) {
    return (point_t){pos.x + dx * (col - HALF), pos.y + dy * (col - HALF)};
}

vector_t find_relative_points(int type, board_t board, point_t pos, int dx, int dy) {
    int id = board[pos.x][pos.y];
    vector_t vec = vector_new(point_t, NULL);
    if (!id) return vec;
    memo_t* memo = id == 1 ? &forbid : &no_forbid;
    int seg_value = encode_segment(get_segment(board, pos, dx, dy));
    int *col, size;
    switch (type) {
        case UPGRADE:
            col = memo->upgrade_col[seg_value], size = memo->upgrade_col_cnt[seg_value];
            break;
        case CONSIST:
            col = memo->consist_col[seg_value], size = memo->consist_col_cnt[seg_value];
            break;
        case DEFEND:
            col = memo->defend_col[seg_value], size = memo->defend_col_cnt[seg_value];
            break;
        default: return vec;
    }
    for (int i = 0; i < size; i++) {
        point_t np = column_to_point(pos, dx, dy, col[i]);
        if (np.x == pos.x && np.y == pos.y) continue;
        assert(in_board(np));
        vector_push_back(vec, np);
    }
    return vec;
}

int is_forbidden_comp(comp_board_t bd, point_t pos, int id, int depth);

/// @brief get pattern by compressed board {bd} and position {pos}
/// @return pattern type
pattern4_t pattern4_type_comp(comp_board_t board, point_t pos, int depth) {
    int id;
    if (!in_board(pos) || !((id = get(board, pos)))) return PAT4_OTHERS;
    const int8_t mid = WIN_LENGTH - 1;
    int idx[4];
    int piece;
    bool consider_forbid = depth > 0 && id == 1;
    for (int8_t i = 0, dx, dy; i < 4; i++) {
        dx = DIRS[i][0], dy = DIRS[i][1];
        segment_t seg;
        for (int8_t j = -WIN_LENGTH + 1; j < WIN_LENGTH; j++) {
            const point_t np = (point_t){pos.x + dx * j, pos.y + dy * j};
            if (!in_board(np))
                seg.pieces[mid + j] = OPPO_PIECE;
            else if (!((piece = get(board, np)))) {
                seg.pieces[mid + j] = EMPTY_PIECE;
            } else
                seg.pieces[mid + j] = ((piece == id) ? SELF_PIECE : OPPO_PIECE);
        }
        // print_segment(seg);
        int segment_value = encode_segment(seg);
        idx[i] = to_pattern(segment_value, consider_forbid);
        if (depth > 1 && idx[i] >= PAT_A3 && idx[i] <= PAT_A4) {
            int col[2];
            get_upgrade_columns(segment_value, consider_forbid, col, 2);
            for (int j = 0; j < 2; j++) {
                if (col[j] != -1) {
                    const point_t np =
                        (point_t){pos.x + dx * (col[j] - mid), pos.y + dy * (col[j] - mid)};
                    // pattern4_t pat4;
                    // if ((pat4 = is_forbidden_comp(board, np, id, depth - 1))) {
                    if (is_forbidden_comp(board, np, id, depth - 1)) {
                        // log("fallback at (%d, %d) for %s", np.x, np.y, pattern4_typename[pat4]);
                        idx[i] = PAT_EMPTY;
                        break;
                    }
                }
            }
        }
    }
    pattern4_t pat4 = to_pattern4(idx[0], idx[1], idx[2], idx[3], consider_forbid);
    return pat4;
}

/// @brief get pattern after putting a piece of player{id} at {pos}
/// @return pattern type
pattern4_t virtual_pat4type_comp(comp_board_t board, point_t pos, int id, int depth) {
    assert(in_board(pos) && !get(board, pos));
    add(board, pos, id);
    const pattern4_t pat4 = pattern4_type_comp(board, pos, depth);
    minus(board, pos, id);
    return pat4;
}

/// @brief check if {pos} is forbidden for player{id}
/// @return 0 if not forbidden, pattern4 type otherwise.
int is_forbidden_comp(comp_board_t board, point_t pos, int id, int depth) {
    assert(in_board(pos));
    if (id != 1) return 0;
    const pattern4_t pat4 = virtual_pat4type_comp(board, pos, id, depth);
    if (pat4 <= PAT4_WIN) return 0;
    return pat4;
}