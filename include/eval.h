#include "board.h"
#include "pattern.h"

typedef struct {
    point_t pos;
    int8_t dx, dy;
    pattern_t pattern;
} threat_t;

typedef vector_t* threat_storage_t[PAT_TYPE_SIZE];  // vector<threat_t>

long long eval(board_t board, int* score_board);

/// @param storage array of vector<threat_t>*, bind pattern type to storage
void scan_threats(comp_board_t board, int id, threat_storage_t storage);
/// @param return vector<threat_t>
vector_t scan_five_threats(board_t board, int id);
/// @param return vector<threat_t>
vector_t scan_four_threats(board_t board, int id);
/// @param return vector<threat_t>
vector_t find_threats_by_threshold(board_t board, int id, pattern_t threshold);
