#include "board.h"
#include "eval.h"
#include "game.h"

#include <string.h>

point_t trivial_move(game_t game) {
    if (game.count == 0) {
        return (point_t){BOARD_SIZE / 2, BOARD_SIZE / 2};
    }
    board_t board;
    memcpy(board, game.board, sizeof(board));
    vector_t self_5 = find_five_points(board, game.cur_id);
    vector_t oppo_5 = find_five_points(board, 3 - game.cur_id);
    if (self_5.size) {
        point_t pos = vector_get(point_t, self_5, 0);
        vector_free_impl(&self_5), vector_free_impl(&oppo_5);
        return pos;
    }
    if (oppo_5.size) {
        for_all_elements(point_t, oppo_5, pos) {
            if (!is_forbidden(board, pos, game.cur_id, false)) {
                vector_free_impl(&self_5), vector_free_impl(&oppo_5);
                return pos;
            }
        }
    }
    vector_free_impl(&self_5), vector_free_impl(&oppo_5);
    return (point_t){-1, -1};
}