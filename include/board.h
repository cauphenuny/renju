#ifndef BOARD_H
#define BOARD_H

#include <stdbool.h>
#include <stdint.h>

#define BOARD_SIZE 15

#define WIN_LENGTH 5
#define PATTERN_LEN  9      // 2 * WIN_LENGTH - 1
#define PATTERN_SIZE 19683  // 3 ** PATTERN_LEN

enum {
    EMPTY_POS,
    SELF_POS,
    OPPO_POS,
};

enum {
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
    PAT_SIZE,
};

enum {
    PAT4_OTHERS,
    PAT4_43,
    PAT4_WIN,
    PAT4_A33,
    PAT4_44,
    PAT4_TL,
    PAT4_SIZE,
};

extern const char* pattern_typename[PAT_SIZE];

extern const char* pattern4_typename[PAT4_SIZE];

typedef struct {
    int8_t x, y;
} point_t;

typedef struct {
    int data[PATTERN_LEN];
    int val;
} segment_t;

typedef int board_t[BOARD_SIZE][BOARD_SIZE];

void emph_print(const board_t, point_t pos);
void print(const board_t);
void wrap_area(const board_t, point_t*, point_t*, int8_t);
bool inboard(point_t);
bool available(board_t board, point_t pos);
void put(board_t board, int id, point_t pos);
int  check_draw(const board_t);
int  check(const board_t, point_t);
int is_banned(const board_t, point_t, int);
void test_ban(void);
int to_pattern(int);
int to_pattern4(int, int, int, int);
void pattern_init();
int segment_encode(segment_t s);
segment_t segment_decode(int v);
void get_critical_column(int pattern, int* danger_col, int limit);

#endif