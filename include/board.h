#ifndef BOARD_H
#define BOARD_H

#include <stdbool.h>
#include <stdint.h>

#define BOARD_SIZE 15
#define BOARD_AREA (BOARD_SIZE * BOARD_SIZE)

#define WIN_LENGTH 5

typedef enum {
    EMPTY_PIECE,
    SELF_PIECE,
    OPPO_PIECE,
    PIECE_SIZE = 4,
} piece_t;

typedef struct {
    int8_t x, y;
} point_t;

typedef int board_t[BOARD_SIZE][BOARD_SIZE];
typedef float fboard_t[BOARD_SIZE][BOARD_SIZE]; // board with float type
typedef float (*pfboard_t)[BOARD_SIZE];         // same as fboard_t but is pointer rather than array

typedef struct {
    point_t* points;
    int size;
} point_array_t;

void print_all(const board_t board, point_t emph_pos, const fboard_t prob);
void print_emph(const board_t board, point_t emph_pos);
void print_prob(const board_t board, const fboard_t prob);
void print(const board_t board);

void wrap_area(const board_t board, point_t* begin, point_t* end, int8_t margin);

#define in_board(pos) (pos.x >= 0 && pos.x < BOARD_SIZE && pos.y >= 0 && pos.y < BOARD_SIZE)
#define in_area(pos, begin, end) \
    (pos.x >= begin.x && pos.x < end.x && pos.y >= begin.y && pos.y < end.y)
bool available(const board_t board, point_t pos);
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

#endif
