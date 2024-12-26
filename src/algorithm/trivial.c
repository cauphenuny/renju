#include "board.h"
#include "eval.h"
#include "game.h"
#include "util.h"
#include "vct.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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
point_t trivial_move(board_t board, int self_id, double time_limit, bool use_vct) {
    const int oppo_id = 3 - self_id;
    vector_t self_5 = vector_new(threat_t, NULL);
    vector_t self_4 = vector_new(threat_t, NULL);
    scan_threats(board, self_id, self_id,
                 (threat_storage_t){[PAT_WIN] = &self_5, [PAT_A4] = &self_4});
    vector_t oppo_5 = vector_new(threat_t, NULL);
    scan_threats(board, oppo_id, oppo_id, (threat_storage_t){[PAT_WIN] = &oppo_5});
    point_t pos = {-1, -1};
    bool is_attack;
    if (self_5.size) {
        threat_t attack = vector_get(threat_t, self_5, 0);
        pos = attack.pos, is_attack = true;
    }
    if (!in_board(pos) && oppo_5.size) {
        for_each(threat_t, oppo_5, defend) {
            if (!is_forbidden(board, defend.pos, self_id, 3)) {
                pos = defend.pos, is_attack = false;
            }
        }
    }
    if (!in_board(pos) && self_4.size) {
        threat_t attack = vector_get(threat_t, self_4, 0);
        pos = attack.pos, is_attack = true;
    }
    vector_free(self_5), vector_free(oppo_5), vector_free(self_4);
    if (in_board(pos)) {
        log_l("%s %c%d", is_attack ? "attack" : "defend", READABLE_POS(pos));
        return pos;
    }

    if (use_vct) {
        double start_time = record_time();
        vector_t vct_sequence = vct(false, board, self_id, time_limit);
        if (vct_sequence.size) {
            pos = vector_get(point_t, vct_sequence, 0);
            log_l("found VCT", get_time(start_time));
            print_points(vct_sequence, PROMPT_NOTE, " -> ");
            // sleep(1);
        }
        vector_free(vct_sequence);
    }
    return pos;
}