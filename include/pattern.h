#ifndef PATTERN_H
#define PATTERN_H

#include "board.h"

#include <stdbool.h>
#include <stdint.h>

#define SEGMENT_LEN  9       // 2 * WIN_LENGTH - 1
#define PATTERN_SIZE 262144  // PIECE_SIZE ** SEGMENT_LEN

typedef enum {
    PAT_ETY,  // empty
    PAT_44,   // double 4  x o o o o . o o o
    PAT_ATL,  // almost tl . o o . o o o . .
    PAT_TL,   // too long  . o o o o o o . .
    PAT_D1,   // dead 1    . . . x o . . . .
    PAT_A1,   // alive 1   . . . . o . . . .
    PAT_D2,   // dead 2    . . x o o . . . .
    PAT_A2,   // alive 2   . . . o o . . . .
    PAT_D3,   // dead 3    . x o o o . . . .
    PAT_A3,   // alive 3   . . o o o . . . .
    PAT_D4,   // dead 4    . o o o o x . . .
    PAT_A4,   // alive 4   . o o o o . . . .
    PAT_5,    // 5         . o o o o o . . .
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

int segment_encode(segment_t s);
segment_t segment_decode(int v);
void segment_print(segment_t s);

pattern_t to_pattern(int segment_value);
pattern4_t to_pattern4(int x, int y, int u, int v);
void pattern_init(void);
void get_upgrade_columns(int segment_value, int* cols, int limit);
void get_patterns(const board_t board, point_t pos, pattern_t arr[]);

#endif
