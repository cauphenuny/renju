#include "eval.h"

#include "board.h"
#include "pattern.h"

#include <stdlib.h>
#include <string.h>

void flatten(board_t board, int perspective, int dir_x, int dir_y, int pieces[], int id2x[],
             int id2y[], int capacity, int* ret_tot) {
    board_t visited = {0};
    memset(pieces, 0, sizeof(int) * capacity);
    int tot = 0;
    for (int x = 0; x < BOARD_SIZE; x++) {
        for (int y = 0; y < BOARD_SIZE; y++) {
            if (visited[x][y]) continue;
            point_t pos = {x, y};
            while (in_board(pos)) {
                id2x[tot] = pos.x;
                id2y[tot] = pos.y;
                visited[pos.x][pos.y] = 1;
                if (!board[pos.x][pos.y]) {
                    pieces[tot++] = EMPTY_PIECE;
                } else if (board[pos.x][pos.y] == perspective) {
                    pieces[tot++] = SELF_PIECE;
                } else {
                    pieces[tot++] = OPPO_PIECE;
                }
                pos.x += dir_x;
                pos.y += dir_y;
            }
            id2x[tot] = id2y[tot] = -1;  // wall cell
            pieces[tot++] = OPPO_PIECE;
        }
    }
    *ret_tot = tot;
}

void scan_threats(board_t board, int id, threat_storage_t storage) {
    int pieces[BOARD_AREA * 2], tot, id2x[BOARD_AREA * 2], id2y[BOARD_AREA * 2];
    vector_t forbidden_pos = vector_new(point_t, NULL);
    for_all_dir(d, dir_x, dir_y) {
        // log("============= dir: {%d, %d} =============", dir[d][0], dir[d][1]);
        flatten(board, id, dir_x, dir_y, pieces, id2x, id2y, BOARD_AREA * 2, &tot);
        // log("tot: %d", tot);
        int value = 0;
        for (int i = 0; i < SEGMENT_LEN; i++) {
            value = value * PIECE_SIZE + OPPO_PIECE;
        }
        for (int idx = 0; idx < tot; idx++) {
            value = ((value * PIECE_SIZE) % SEGMENT_MASK) + pieces[idx];
            pattern_t pat = to_upgraded_pattern(value, id == 1);
            // log("pat: %s", pattern_typename[pat]);
            if (storage[pat]) {
                int columns[5];
                get_upgrade_columns(value, id == 1, columns, 5);
                for (int j = 0; j < 5; j++) {
                    if (columns[j] != -1) {
                        int cur_idx = idx - columns[j];
                        point_t pos = {id2x[cur_idx], id2y[cur_idx]};
                        board[pos.x][pos.y] = id;
                        pattern_t real_pat = to_pattern(
                            encode_segment(get_segment(board, pos, dir_x, dir_y)), id == 1);
                        board[pos.x][pos.y] = 0;
                        if (real_pat != pat) continue;  // to solve the problem of - o - o o - o - -
                        bool save = true;
                        for_each(threat_t, *storage[pat], stored_threat) {
                            if (stored_threat.id == id && point_equal(stored_threat.pos, pos) &&
                                stored_threat.dir.x == dir_x && stored_threat.dir.y == dir_y) {
                                save = false;
                                break;
                            }
                        }
                        for_each(point_t, forbidden_pos, f_pos) {
                            if (f_pos.x == pos.x && f_pos.y == pos.y) {
                                save = false;
                                break;
                            }
                        }
                        if (save && is_forbidden(board, pos, id, 3)) {
                            save = false;
                            vector_push_back(forbidden_pos, pos);
                        }
                        if (save) {
                            threat_t threat = {
                                .pos = pos,
                                .dir =
                                    {
                                        .x = dir_x,
                                        .y = dir_y,
                                    },
                                .pattern = pat,
                                .id = id,
                            };
                            vector_push_back(*storage[pat], threat);
                        }
                    }
                }
            }
        }
    }
    vector_free(forbidden_pos);
}

vector_t scan_four_threats(board_t board, int id) {
    vector_t result = vector_new(threat_t, NULL);
    threat_storage_t storage = {
        [PAT_A4] = &result,
        [PAT_D4] = &result,
    };
    scan_threats(board, id, storage);
    return result;
}

vector_t scan_five_threats(board_t board, int id) {
    vector_t result = vector_new(threat_t, NULL);
    threat_storage_t storage = {
        [PAT_WIN] = &result,
    };
    scan_threats(board, id, storage);
    return result;
}

vector_t scan_threats_by_threshold(board_t board, int id, pattern_t threshold) {
    vector_t result = vector_new(threat_t, NULL);
    threat_storage_t storage = {0};
    for (int i = (int)threshold; i < (int)PAT_TYPE_SIZE; i++) {
        storage[i] = &result;
    }
    scan_threats(board, id, storage);
    return result;
}

long long eval_pos(board_t board, point_t pos) {
    int id = board[pos.x][pos.y];
    long long score_board[PAT_TYPE_SIZE] = {
        [PAT_EMPTY] = 0, [PAT_44] = 0,     [PAT_ATL] = 0,      [PAT_TL] = 0,   [PAT_D1] = 5,
        [PAT_A1] = 10,   [PAT_D2] = 10,    [PAT_A2] = 150,     [PAT_D3] = 100, [PAT_A3] = 1000,
        [PAT_D4] = 1000, [PAT_A4] = 10000, [PAT_WIN] = 100000,
    };
    long long relativity[PAT_TYPE_SIZE][PAT_TYPE_SIZE] = {
        [PAT_A3] = {[PAT_A3] = score_board[PAT_A4] / 2, [PAT_D4] = score_board[PAT_A4]},
        [PAT_D4] = {[PAT_A3] = score_board[PAT_A4], [PAT_D4] = score_board[PAT_A4] / 2},
        [PAT_A2] = {[PAT_A2] = score_board[PAT_A3] / 5, [PAT_D3] = score_board[PAT_A3] / 5},
        [PAT_D3] = {[PAT_A2] = score_board[PAT_A3] / 5, [PAT_D3] = score_board[PAT_A3] / 5},
    };
    long long result = 0;
    int cnt[PAT_TYPE_SIZE] = {0};
    for_all_dir(d, dx, dy) {
        pattern_t pat = to_pattern(encode_segment(get_segment(board, pos, dx, dy)), id == 1);
        cnt[pat]++;
    }
    for (int i = 0; i < PAT_TYPE_SIZE; i++) {
        result += score_board[i] * cnt[i];
    }
    for (int i = 0; i < PAT_TYPE_SIZE; i++) {
        if (cnt[i] >= 2) result += relativity[i][i];
        for (int j = i + 1; j < PAT_TYPE_SIZE; j++) {
            if (cnt[i] && cnt[j]) {
                result += relativity[i][j];
            }
        }
    }
    return result * (id == 1 ? 1 : -1);
}

long long add_with_eval(board_t board, long long current_eval, point_t pos, int id) {
    for_all_dir(d, dx, dy) {
        for (int offset = -SEGMENT_LEN / 2; offset <= SEGMENT_LEN / 2; offset++) {
            int x = pos.x + dx * offset, y = pos.y + dy * offset;
            point_t np = {x, y};
            if (in_board(np) && board[x][y]) {
                current_eval -= eval_pos(board, np);
            }
        }
    }
    board[pos.x][pos.y] = id;
    current_eval += eval_pos(board, pos);
    for_all_dir(d, dx, dy) {
        for (int offset = -SEGMENT_LEN / 2; offset <= SEGMENT_LEN / 2; offset++) {
            if (offset == 0) continue;
            int x = pos.x + dx * offset, y = pos.y + dy * offset;
            point_t np = {x, y};
            if (in_board(np) && board[x][y]) {
                current_eval += eval_pos(board, (point_t){x, y});
            }
        }
    }
    return current_eval;
}

long long eval(board_t board, int* score_board) {
    // clang-foramt off
    (void)score_board;
    long long result = 0;
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (board[i][j]) {
                result += eval_pos(board, (point_t){i, j});
            }
        }
    }
    return result;
}
