#include "board.h"
#include "eval.h"
#include "game.h"
#include "threat.h"
#include "util.h"

#include <stdlib.h>
#include <string.h>

/// @brief generate a available random move from {game.board}
point_t random_move(game_t game) {
    point_t points[BOARD_AREA];
    int tot = 0;
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
        point_t pos = (point_t){i, j};
        if (game.board[i][j]) continue;
        if (is_forbidden(game.board, pos, game.cur_id, -1)) continue;
        points[tot++] = pos;
        }
    }
    return points[rand() % tot];
}

/// @brief find a trivial move from {game.board}
point_t trivial_move(game_t game, bool use_vct) {
    if (game.count == 0) {
        return (point_t){BOARD_SIZE / 2, BOARD_SIZE / 2};
    }
    board_t board;
    memcpy(board, game.board, sizeof(board));
    vector_t self_5 = vector_new(threat_t, NULL);
    vector_t self_4 = vector_new(threat_t, NULL);
    scan_threats(board, game.cur_id, (threat_storage_t){[PAT_WIN] = &self_5, [PAT_A4] = &self_4});
    vector_t oppo_5 = scan_five_threats(board, 3 - game.cur_id);
    point_t pos = {-1, -1};
    if (self_5.size) {
        threat_t attack = vector_get(threat_t, self_5, 0);
        pos = attack.pos;
    }
    if (!in_board(pos) && oppo_5.size) {
        for_each(threat_t, oppo_5, defend) {
            if (!is_forbidden(board, defend.pos, game.cur_id, 3)) {
                pos = defend.pos;
            }
        }
    }
    if (!in_board(pos) && self_4.size) {
        threat_t attack = vector_get(threat_t, self_4, 0);
        pos = attack.pos;
    }
    vector_free(self_5), vector_free(oppo_5), vector_free(self_4);
    if (in_board(pos)) return pos;

    if (use_vct) {
        double start_time = record_time();
        vector_t vct_sequence = vct(false, game.board, game.cur_id, game.time_limit / 20.0);
        if (vct_sequence.size) {
            pos = vector_get(point_t, vct_sequence, 0);
            log("found VCT in %.2lfms", get_time(start_time));
            print_points(vct_sequence, PROMPT_NOTE, "->");
            // if (vcf_sequence.size > 2) {
            //     prompt_pause();
            // }
        }
        vector_free(vct_sequence);
    }
    return pos;
}