#include "board.h"
#include "pattern.h"

typedef vector_t* point_storage_t[PAT_TYPE_SIZE];

void find_points(comp_board_t board, int id, point_storage_t storage);
vector_t find_five_points(board_t board, int id);
vector_t find_four_points(board_t board, int id);
vector_t find_critical_points(board_t board, int id, pattern_t threshold);
long long eval(board_t board, int* score_board);