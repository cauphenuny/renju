#ifndef PATTERN_H
#define PATTERN_H

#include "board.h"

#include <stdbool.h>
#include <stdint.h>

// #define SEGMENT_LEN       9       // 2 * WIN_LENGTH + 1
// #define SEGMENT_REAL_SIZE 19683   // max(EMPTY_PIECE, SELF_PIECE, OPPO_PIECE) ** SEGMENT_LEN
// #define SEGMENT_SIZE      262144  // PIECE_SIZE ** SEGMENT_LEN

#define SEGMENT_LEN       11       // 2 * WIN_LENGTH + 1
#define SEGMENT_REAL_SIZE 177147   // max(EMPTY_PIECE, SELF_PIECE, OPPO_PIECE) ** SEGMENT_LEN
#define SEGMENT_SIZE      4194304  // PIECE_SIZE ** SEGMENT_LEN

#define HALF (SEGMENT_LEN / 2)

#if (1 << (SEGMENT_LEN * 2)) != SEGMENT_SIZE
#    error "SEGMENT_SIZE is not equal to 4 ** SEGMENT_LEN"
#endif

typedef enum {
    PAT_EMPTY,  // empty
    PAT_44,     // double 4  . x o o o o . o o o .
    PAT_ATL,    // almost tl . . o o . o o o . . .
    PAT_TL,     // too long  . . o o o o o o . . .
    PAT_D1,     // dead 1    . . . . x o . . . . .
    PAT_A1,     // alive 1   . . . . . o . . . . .
    PAT_D2,     // dead 2    . . . x o o . . . . .
    PAT_A2,     // alive 2   . . . . o o . . . . .
    PAT_D3,     // dead 3    . . x o o o . . . . .
    PAT_A3,     // alive 3   . . . o o o . . . . .
    PAT_D4,     // dead 4    . . o o o o x . . . .
    PAT_A4,     // alive 4   . . o o o o . . . . .
    PAT_WIN,    // 5         . . o o o o o . . . .
    PAT_TYPE_SIZE,
} pattern_t;

typedef enum {
    PAT4_OTHERS,
    PAT4_WIN,
    PAT4_A33,
    PAT4_44,
    PAT4_TL,
    PAT4_TYPE_SIZE,
} pattern4_t;

extern const char* pattern_typename[PAT_TYPE_SIZE];
extern const char* pattern4_typename[PAT4_TYPE_SIZE];

typedef struct {
    piece_t pieces[SEGMENT_LEN];  // something like . . o . . x . o .
} segment_t;

int encode_segment(segment_t s);
segment_t decode_segment(int v, int base);
void print_segment(segment_t s, bool consider_forbid);
bool segment_valid(segment_t s);
segment_t get_segment(board_t board, point_t pos, int dx, int dy, int id);
enum {
    ATTACK,
    DEFENSE,
};

vector_t find_relative_points(int type, board_t board, point_t pos, int dx, int dy, int id, bool put_piece);

pattern_t to_pattern(int segment_value, bool consider_forbid);
pattern_t get_pattern(board_t board, point_t pos, int dx, int dy, int self_id);
pattern4_t get_pattern4(board_t board, point_t pos, int self_id, bool put_piece);
pattern_t to_upgraded_pattern(int segment_value, bool consider_forbid);
pattern4_t to_pattern4(int x, int y, int u, int v, bool consider_forbid);
void pattern_init(void);
void get_attack_columns(int segment_value, bool consider_forbid, int* cols, int limit);
point_t column_to_point(point_t pos, int dx, int dy, int col);

pattern4_t pattern4_type_comp(comp_board_t board, point_t pos, int depth);
pattern4_t virtual_pat4type_comp(comp_board_t board, point_t pos, int id, int depth);
int is_forbidden_comp(comp_board_t bd, point_t pos, int id, int depth);

#endif
