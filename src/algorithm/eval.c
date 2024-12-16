#include "eval.h"

#include "board.h"
#include "pattern.h"

#include <stdlib.h>
#include <string.h>

void flatten(comp_board_t board, int perspective, int dir_x, int dir_y, int pieces[], int id2x[],
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
                if (!get(board, pos)) {
                    pieces[tot++] = EMPTY_PIECE;
                } else if (get(board, pos) == perspective) {
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

void scan_threats(comp_board_t board, int id, threat_storage_t storage) {
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
                        bool save = true;
                        for_each(threat_t, *storage[pat], stored_threat) {
                            if (stored_threat.pos.x == pos.x && stored_threat.pos.y == pos.y &&
                                stored_threat.dx == dir_x && stored_threat.dy == dir_y) {
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
                        if (save && is_forbidden_comp(board, pos, id, 3)) {
                            save = false;
                            vector_push_back(forbidden_pos, pos);
                        }
                        if (save) {
                            threat_t threat = {
                                .pos = pos,
                                .dx = dir_x,
                                .dy = dir_y,
                                .pattern = pat,
                            };
                            vector_push_back(*storage[pat], threat);
                        }
                    }
                }
            }
        }
    }
    vector_free(&forbidden_pos);
}

vector_t scan_four_threats(board_t board, int id) {
    comp_board_t cpboard;
    encode(board, cpboard);
    vector_t result = vector_new(threat_t, NULL);
    threat_storage_t storage = {
        [PAT_A4] = &result,
        [PAT_D4] = &result,
    };
    scan_threats(cpboard, id, storage);
    return result;
}

vector_t scan_five_threats(board_t board, int id) {
    comp_board_t cpboard;
    encode(board, cpboard);
    vector_t result = vector_new(threat_t, NULL);
    threat_storage_t storage = {
        [PAT_WIN] = &result,
    };
    scan_threats(cpboard, id, storage);
    return result;
}

vector_t scan_threats_by_threshold(board_t board, int id, pattern_t threshold) {
    comp_board_t cpboard;
    encode(board, cpboard);
    vector_t result = vector_new(threat_t, NULL);
    threat_storage_t storage = {0};
    for (int i = (int)threshold; i < (int)PAT_TYPE_SIZE; i++) {
        storage[i] = &result;
    }
    scan_threats(cpboard, id, storage);
    return result;
}

long long eval(board_t board, int* score_board) {
    int pieces[BOARD_AREA * 2], tot, id2x[BOARD_AREA * 2], id2y[BOARD_AREA * 2];
    comp_board_t cpboard;
    encode(board, cpboard);
    long long result = 0;
    // clang-foramt off
    int default_score_board[PAT_TYPE_SIZE] = {
        [PAT_ETY] = 0,        [PAT_44] = 0,     [PAT_ATL] = 0,    [PAT_TL] = 0,
        [PAT_D1] = 5,         [PAT_A1] = 10,    [PAT_D2] = 500,   [PAT_A2] = 1000,
        [PAT_D3] = 10000,     [PAT_A3] = 50000, [PAT_D4] = 50000, [PAT_A4] = 1000000,
        [PAT_WIN] = 20000000,
    };
    if (!score_board) score_board = default_score_board;
    // clang-format on
    for_all_dir(d, dx, dy) {
        flatten(cpboard, 1, dx, dy, pieces, id2x, id2y, BOARD_AREA * 2, &tot);
        int value = 0;
        for (int i = 0; i < SEGMENT_LEN; i++) {
            value = value * PIECE_SIZE + OPPO_PIECE;
        }
        for (int idx = 0; idx < tot; idx++) {
            value = ((value * PIECE_SIZE) % SEGMENT_MASK) + pieces[idx];
            pattern_t pat = to_pattern(value, true);
            // log("pat: %s", pattern_typename[pat]);
            result += score_board[pat];
        }
    }
    for_all_dir(d, dx, dy) {
        flatten(cpboard, 2, dx, dy, pieces, id2x, id2y, BOARD_AREA * 2, &tot);
        int value = 0;
        for (int i = 0; i < SEGMENT_LEN; i++) {
            value = value * PIECE_SIZE + OPPO_PIECE;
        }
        for (int idx = 0; idx < tot; idx++) {
            value = ((value * PIECE_SIZE) % SEGMENT_MASK) + pieces[idx];
            pattern_t pat = to_pattern(value, false);
            // log("pat: %s", pattern_typename[pat]);
            result -= score_board[pat];
        }
    }
    return result;
}
