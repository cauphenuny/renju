#ifndef BOARD_H
#define BOARD_H

#include <stdbool.h>
#include <stdint.h>

#define BOARD_SIZE 15

#define WIN_LENGTH 5
#define SEGMENT_LEN  9      // 2 * WIN_LENGTH - 1
#define PATTERN_SIZE 19683  // 3 ** SEGMENT_LEN

typedef enum {
    EMPTY_PIECE,
    SELF_PIECE,
    OPPO_PIECE,
    PIECE_SIZE,
} piece_t;

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
    PAT4_43,
    PAT4_WIN,
    PAT4_A33,
    PAT4_44,
    PAT4_TL,
    PAT4_TYPE_SIZE,
} pattern4_t;

extern const char* pattern_typename[PAT_TYPE_SIZE];

extern const char* pattern4_typename[PAT4_TYPE_SIZE];

typedef struct {
    int8_t x, y;
} point_t;

typedef int board_t[BOARD_SIZE][BOARD_SIZE];
typedef double fboard_t[BOARD_SIZE][BOARD_SIZE]; // board with float type
typedef double (*pfboard_t)[BOARD_SIZE];         // same as fboard_t but is pointer rather than array

void print_implement(const board_t board, point_t emph_pos, const fboard_t prob);
void emphasis_print(const board_t board, point_t emph_pos);
void probability_print(const board_t board, const fboard_t prob);
void print(const board_t board);

void wrap_area(const board_t board, point_t* begin, point_t* end, int8_t margin);

#define in_board(pos) (pos.x >= 0 && pos.x < BOARD_SIZE && pos.y >= 0 && pos.y < BOARD_SIZE)
#define in_area(pos, begin, end) \
    (pos.x >= begin.x && pos.x < end.x && pos.y >= begin.y && pos.y < end.y)
bool available(const board_t board, point_t pos);
int is_forbidden(const board_t board, point_t pos, int id, bool enable_log);

void put(board_t board, int id, point_t pos);

int check(const board_t board, point_t pos);
bool is_draw(const board_t board);
bool have_space(const board_t board, int id);
bool is_equal(const board_t b1, const board_t b2);

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
int evaluate_pos(const board_t board, point_t pos, int id, bool check_forbid);

#endif
