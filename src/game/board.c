#include "board.h"

#include "pattern.h"
#include "util.h"

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define printf(...) log_s(__VA_ARGS__)

/// @brief deserialize a board from a string representation {str}, save it to {dest}
void board_deserialize(board_t dest, const char* str) {
    memset(dest, 0, sizeof(board_t));
    int x = 0, y = 0, l = strlen(str);
    for (int i = 0; i < l; i++) {
        switch (str[i]) {
            case '.': y++; break;
            case 'o': dest[x][++y] = 1; break;
            case 'x': dest[x][++y] = 2; break;
            case '\\': y = 0, x++;
        }
    }
}

/// @brief serialize {board} to a string representation {dest}
void board_serialize(const board_t board, char* dest) {
    int p = 0;
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            switch (board[i][j]) {
                case 0: dest[p++] = '.'; break;
                case 1: dest[p++] = 'o'; break;
                case 2: dest[p++] = 'x'; break;
            }
            dest[p++] = ' ';
        }
        dest[p++] = '\n';
    }
    dest[p] = '\0';
}

static bool point_vector_contains(vector_t vec, point_t pos) {
    return vector_contains(point_t, vec, pos);
}

int point_serialize(char* dest, size_t size, const void* ptr) {
    const point_t* point = ptr;
    return snprintf(dest, size, "%c%d", READABLE_POS(*point));
}

void print_points(vector_t point_array, int log_level, const char* delim) {
    char buffer[1024];
    vector_serialize(buffer, 1024, delim, point_array, point_serialize);
    log_add(log_level, "%s", buffer);
}

static char* piece_color[] = {RESET, GREEN, RED, L_GREEN, L_RED};

void set_color(bool green_first) {
    if (green_first) {
        piece_color[1] = GREEN, piece_color[3] = L_GREEN;
        piece_color[2] = RED, piece_color[4] = L_RED;
    } else {
        piece_color[1] = RED, piece_color[3] = L_RED;
        piece_color[2] = GREEN, piece_color[4] = L_GREEN;
    }
}

void print_impl(const board_t board, vector_t emph_pos, const fboard_t prob) {
    if (log_locked() || log_disabled()) return;
    const char* border_line[3][5] = {{"╔", "═", "╤", "═", "╗"},   //
                                     {"╟", "─", "┼", "─", "╢"},   //
                                     {"╚", "═", "╧", "═", "╝"}};  //
    const char* piece_ch[5] = {
        " ",  // empty
        "o",  // previous p1 pieces
        "x",  // previous p2 pieces
        "o",  // current p1 piece
        "x",  // current p2 piece
    };
#define LEVELS 7
    const struct {
        double thresh;
        const char* ch;
    } levels[LEVELS] = {
        {0.01, DARK "#" RESET},             // 0.01 < p <= 0.05
        {0.05, "#"},                        // 0.05 < p <= 0.20
        {0.10, BOLD "#" RESET},             // 0.05 < p <= 0.20
        {0.20, L_YELLOW "#" RESET},         // 0.20 < p <= 0.50
        {0.50, L_RED "#" RESET},            // 0.50 < p <= 0.85
        {0.85, UNDERLINE L_RED "#" RESET},  // 0.85 < p <= 1
        {1.00, "?"},                        // invalid
    };
    for (int i = BOARD_SIZE - 1, line_type, col_type; i >= 0; i--) {
        switch (i) {
            case BOARD_SIZE - 1: line_type = 0; break;
            case 0: line_type = 2; break;
            default: line_type = 1;
        }
#if DEBUG_LEVEL >= 2
        printf("%2d%c", i, " ["[point_vector_contains(emph_pos, (point_t){i, 0})]);
#else
        printf("%2d%c", i + 1, " ["[point_vector_contains(emph_pos, (point_t){i, 0})]);
#endif
        for (int j = 0; j < 2 * BOARD_SIZE - 1; j++) {
            switch (j) {
                case 0: col_type = 0; break;
                case 2 * BOARD_SIZE - 2: col_type = 4; break;
                default: col_type = 2 + (j & 1);
            }
            if (j & 1) {
                int flag = (int)point_vector_contains(emph_pos, (point_t){i, j / 2}) +
                           2 * (int)point_vector_contains(emph_pos, (point_t){i, (j + 1) / 2});
                switch (flag) {
                    case 0b00: printf(DARK "%s" RESET, border_line[line_type][col_type]); break;
                    case 0b01: printf("]"); break;
                    case 0b10: printf("["); break;
                    case 0b11: printf("|"); break;
                }
            } else if (board[i][j / 2]) {
                int p = board[i][j / 2];
                printf("%s%s%s", piece_color[p], piece_ch[p], RESET);
            } else if (prob && prob[i][j / 2] > levels[0].thresh) {
                int level = 0;
                for (int k = 0; k < LEVELS; k++) {
                    if (prob[i][j / 2] > levels[k].thresh) level = k;
                }
                printf("%s", levels[level].ch);
            } else {
                printf(DARK "%s" RESET, border_line[line_type][col_type]);
            }
        }
        printf("%c\n", " ]"[point_vector_contains(emph_pos, (point_t){i, BOARD_SIZE - 1})]);
    }
    printf("   ");
    for (int i = 0; i < 2 * BOARD_SIZE - 1; i++) {
#if DEBUG_LEVEL >= 2
        if (!(i & 1)) printf("%-2d", i / 2);
#else
        printf("%c", (i & 1) ? ' ' : 'A' + i / 2);
#endif
    }
    printf("\n");
#undef LEVELS
}

/// @brief print {board} with one emphasized position {pos} or predicted probability
void print_all(const board_t board, point_t emph_pos, const fboard_t prob) {
    vector_t points = vector_new(point_t, NULL);
    vector_push_back(points, emph_pos);
    print_impl(board, points, prob);
    vector_free(points);
}

/// @brief print {board} with one emphasized position
void print_emph(const board_t board, point_t emph_pos) { print_all(board, emph_pos, NULL); }

/// @brief print {board} predicted probability
void print_prob(const board_t board, const fboard_t prob) {
    print_all(board, (point_t){-1, -1}, prob);
}

void print_emph_mutiple(const board_t board, vector_t points) {
    fboard_t prob;
    memset(prob, 0, sizeof(prob));
    print_impl(board, points, prob);
}

/// @brief print {board} without emphasis
void print(const board_t board) { return print_emph(board, (point_t){-1, -1}); }

/// @brief get a minimal area [{begin}, {end}) that wraps up pieces in {board}
/// @param begin left-top corner of wrapped area
/// @param end right-bottom corner of wrapped area
/// @param margin min margin
void wrap_area(const board_t board, point_t* begin, point_t* end, int8_t margin) {
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
bool available(const board_t board, point_t pos) {
    if (!in_board(pos) || board[pos.x][pos.y]) return false;
    return true;
}

/// @brief put a player{id}'s piece at {pos} on {board}
void put(board_t board, int id, point_t pos) { board[pos.x][pos.y] = id; }

/// @brief check if {board} is a draw situation
bool is_draw(const board_t board) {
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (!board[i][j]) return false;
        }
    }
    return true;
}

/// @brief check if {b1} is equal to {b2} at every position
bool is_equal(const board_t b1, const board_t b2) {
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
bool have_space(const board_t board, int id) {
    board_t dyn_board;
    memcpy(dyn_board, board, sizeof(board_t));
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (!board[i][j] && !is_forbidden(dyn_board, (point_t){i, j}, id, 3)) {
                return true;
            }
        }
    }
    // print(board), prompt_pause();
    return false;
}

/// @brief check if there is a winner on {board} at {pos}
/// @param board current board
/// @param pos position to check
/// @return 1 for player 1 win, -1 for player 2 win, 0 for no winner
int check(board_t board, point_t pos) {
    int id = board[pos.x][pos.y];
    pattern4_t pat4 = get_pattern4(board, pos, id, false);
    return pat4 == PAT4_WIN ? (id == 1 ? 1 : -1) : 0;
}

int put_and_check(board_t board, point_t pos, int id) {
    board[pos.x][pos.y] = id;
    int result = check(board, pos);
    board[pos.x][pos.y] = 0;
    return result;
}

/// @brief check if {pos} is forbidden for player{id}
/// @param board current board
/// @param pos position to check
/// @param id current id
/// @param max_depth max depth, -1 if no limit
/// @return 0 for accept, pat4 if forbidden
bool enable_forbid_log;
int is_forbidden(board_t board, point_t pos, int id, int max_depth) {
    assert(in_board(pos) && !board[pos.x][pos.y]);
    if (max_depth == 0) return 0;
    if (id != 1) return 0;
    // print(board);
    pattern_t pats[4] = {PAT_EMPTY, PAT_EMPTY, PAT_EMPTY, PAT_EMPTY};
    pattern4_t pat4;
    segment_t seg[4];
    int value[4];

    // print_emph(board, pos);
    // pause();
    board[pos.x][pos.y] = id;
    for_all_dir(i, dx, dy) {
        seg[i] = get_segment(board, pos, dx, dy);
        value[i] = encode_segment(seg[i]);
        pats[i] = to_pattern(value[i], true);
    }
    pat4 = to_pattern4(pats[0], pats[1], pats[2], pats[3], true);
    if (pat4 == PAT4_A33) {
        for_all_dir(i, dx, dy) {
            if (pats[i] == PAT_A3) {
                int cols[2];
                get_attack_columns(value[i], true, cols, 2);
                for (int j = 0; j < 2; j++) {
                    if (cols[j] != -1) {
                        const point_t np =
                            (point_t){pos.x + dx * (cols[j] - HALF), pos.y + dy * (cols[j] - HALF)};
                        if (is_forbidden(board, np, id, max_depth > 0 ? max_depth - 1 : -1)) {
                            pats[i] = PAT_EMPTY;  // incorrect, but enough for checking forbid
                            break;
                        }
                    }
                }
            }
        }
        pat4 = to_pattern4(pats[0], pats[1], pats[2], pats[3], true);
    }
    board[pos.x][pos.y] = 0;

    if (pat4 <= PAT4_WIN) return 0;
    if (enable_forbid_log) {
        log_l("forbidden pos, reason: %s", pattern4_typename[pat4]);
        log_l("detailed information:");
        print_emph(board, pos);
        for (int i = 0; i < 4; i++) {
            print_segment(seg[i], true);
        }
    }
    return (int)pat4;
}

/// @brief encode from raw board to compressed board
void encode(const board_t src, comp_board_t dest) {
    for (int i = 0; i < BOARD_SIZE; i++) {
        dest[i] = 0;
        for (int j = BOARD_SIZE - 1; j >= 0; j--) {
            dest[i] = dest[i] * 4 + src[i][j];
        }
    }
}

/// @brief decode from compressed board to raw board
void decode(const comp_board_t src, board_t dest) {
    for (int i = 0; i < BOARD_SIZE; i++) {
        line_t tmp = src[i];
        for (int j = 0; j < BOARD_SIZE; j++) {
            dest[i][j] = tmp & 3;
            tmp >>= 2;
        }
    }
}

void print_compressed_board(const comp_board_t board, point_t emph_pos) {
    board_t b;
    decode(board, b);
    print_emph(b, emph_pos);
}
