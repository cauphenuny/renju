#ifndef BOARD_H
#define BOARD_H

#include <stdbool.h>
#include <stdint.h>

#define BOARD_SIZE 15

#define WIN_LENGTH 5

typedef enum {
    EMPTY_PIECE,
    SELF_PIECE,
    OPPO_PIECE,
    PIECE_SIZE = 4,
} piece_t;

#define SEGMENT_LEN  9      // 2 * WIN_LENGTH - 1
#define PATTERN_SIZE 262144 // PIECE_SIZE ** SEGMENT_LEN

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
int is_forbidden_legacy(const board_t board, point_t pos, int id, bool enable_log);
int is_forbidden(board_t board, point_t pos, int id, bool enable_log);

void put(board_t board, int id, point_t pos);

int check(const board_t board, point_t pos);
bool is_draw(const board_t board);
bool have_space(const board_t board, int id);
bool is_equal(const board_t b1, const board_t b2);

#if BOARD_SIZE <= 16
typedef uint32_t line_t;
#elif BOARD_SIZE <= 32
typedef uint64_t line_t;
#else
#    error "board size too large!"
#endif

typedef line_t comp_board_t[BOARD_SIZE];  // compressed board

/// @brief read and modify compressed board
#define get_xy(arr, x, y)      (int)((arr[x] >> ((y) * 2)) & 3)
#define set_xy(arr, x, y, v)   arr[x] += ((v) - get_xy(arr, x, y)) * (1 << ((y) * 2))
#define add_xy(arr, x, y, v)   arr[x] += (((v) << ((y) * 2)))
#define minus_xy(arr, x, y, v) arr[x] -= (((v) << ((y) * 2)))
#define get(arr, p)            get_xy(arr, p.x, p.y)
#define set(arr, p, v)         set_xy(arr, p.x, p.y, v)
#define add(arr, p, v)         add_xy(arr, p.x, p.y, v)
#define minus(arr, p, v)       minus_xy(arr, p.x, p.y, v)

void encode(const board_t src, comp_board_t dest);
void decode(const comp_board_t src, board_t dest);
void print_compressed_board(const comp_board_t board, point_t emph_pos);

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
