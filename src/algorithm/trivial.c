#include "board.h"
#include "eval.h"
#include "game.h"
#include "threat.h"
#include "util.h"

#include <string.h>

point_t trivial_move(game_t game, bool use_vcf) {
    if (game.count == 0) {
        return (point_t){BOARD_SIZE / 2, BOARD_SIZE / 2};
    }
    board_t board;
    memcpy(board, game.board, sizeof(board));
    vector_t self_5 = scan_five_threats(board, game.cur_id);
    vector_t oppo_5 = scan_five_threats(board, 3 - game.cur_id);
    if (self_5.size) {
        point_t pos = vector_get(point_t, self_5, 0);
        vector_free(&self_5), vector_free(&oppo_5);
        return pos;
    }
    if (oppo_5.size) {
        for_each(point_t, oppo_5, pos) {
            if (!is_forbidden(board, pos, game.cur_id, false)) {
                vector_free(&self_5), vector_free(&oppo_5);
                return pos;
            }
        }
    }
    vector_free(&self_5), vector_free(&oppo_5);

    point_t pos = {-1, -1};
    if (use_vcf) {
        int start_time = record_time();
        vector_t vcf_sequence = vcf(game.board, game.cur_id);
        if (vcf_sequence.size) {
            pos = vector_get(point_t, vcf_sequence, 0);
            log_w("found VCF in %dms", get_time(start_time));
            print_vcf(vcf_sequence);
            // if (vcf_sequence.size > 2) {
            //     prompt_pause();
            // }
        }
        vector_free(&vcf_sequence);
    }
    return pos;
}