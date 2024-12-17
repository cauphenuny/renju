#include "board.h"
#include "pattern.h"

typedef struct {
    point_t pos;
    point_t dir;
    pattern_t pattern;
    int id;
} threat_t;

typedef vector_t* threat_storage_t[PAT_TYPE_SIZE];  // vector<threat_t>

#define EVAL_INF 1000000

long long eval(board_t board, int* score_board);

/// @param storage array of vector<threat_t>*, bind pattern type to storage
void scan_threats(board_t board, int id, threat_storage_t storage);
/// @param return vector<threat_t>
vector_t scan_five_threats(board_t board, int id);
/// @param return vector<threat_t>
vector_t scan_four_threats(board_t board, int id);
/// @param return vector<threat_t>
vector_t find_threats_by_threshold(board_t board, int id, pattern_t threshold);

long long add_with_eval(board_t board, long long current_eval, point_t pos, int id);