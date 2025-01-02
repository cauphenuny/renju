/// @file trivial.c
/// @brief implementation of several trivial strategies

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
typedef struct {
    point_t moves[8];
    int size;
    int score;
    const char* name;
} opening_t;

static const opening_t black_openings[] = {
    {
        .moves =
            {
                {7, 7},
                {6, 8},
                {5, 6},
                {8, 7},
                {7, 8},
                {7, 6},
                {6, 5},
            },
        .size = 7,
        .score = 7,
        .name = "#1-0",
    },
    {
        .moves =
            {
                {7, 7},
                {6, 8},
                {5, 6},
                {6, 7},
                {6, 6},
            },
        .size = 5,
        .score = 3,
        .name = "#1-1",
    },
    {
        .moves =
            {
                {7, 7},
                {6, 8},
                {5, 6},
                {6, 6},
                {6, 5},
            },
        .size = 5,
        .score = 2,
        .name = "#1-2",
    },
    {
        .moves =
            {
                {7, 7},
                {6, 8},
                {5, 6},
                {8, 8},
                {5, 8},
            },
        .size = 5,
        .score = 2,
        .name = "#1-3",
    },
    {
        .moves =
            {
                {7, 7},
                {6, 8},
                {5, 6},
                {6, 9},
                {6, 6},
            },
        .size = 5,
        .score = 2,
        .name = "#1-4",
    },
    {
        .moves =
            {
                {7, 7},
                {7, 8},
                {6, 8},
                {8, 6},
                {6, 9},
                {6, 7},
                {5, 9},
            },
        .size = 7,
        .score = 4,
        .name = "#2-0",
    },
    {
        .moves =
            {
                {7, 7},
                {7, 8},
                {6, 8},
                {6, 7},
                {8, 9},
                {8, 8},
                {5, 9},
            },
        .size = 7,
        .score = 4,
        .name = "#2-1",
    },
    {
        .moves =
            {
                {7, 7},
                {7, 8},
                {6, 8},
                {6, 7},
                {8, 9},
                {8, 6},
                {8, 10},
            },
        .size = 7,
        .score = 4,
        .name = "#2-2",
    },
};

static const opening_t white_openings[] = {
    {
        .moves =
            {
                {7, 7},
                {6, 8},
                {5, 6},
                {8, 7},
                {7, 8},
                {7, 6},
                {6, 7},
                {4, 5},
            },
        .score = 2,
        .name = "#1",
        .size = 8,
    },
    {
        .moves =
            {
                {7, 7},
                {6, 8},
                {5, 7},
                {6, 7},
                {6, 6},
                {5, 5},
            },
        .score = 2,
        .name = "#2",
        .size = 6,
    },
    {
        .moves =
            {
                {7, 7},
                {6, 8},
                {7, 6},
                {7, 5},
            },
        .score = 2,
        .name = "#3",
        .size = 4,
    },
    {
        .moves =
            {
                {7, 7},
                {6, 8},
                {6, 7},
                {7, 6},
                {8, 8},
                {8, 7},
            },
        .score = 2,
        .name = "#3-0",
        .size = 6,
    },
    {
        .moves =
            {
                {7, 7},
                {6, 8},
                {6, 7},
                {7, 6},
                {5, 7},
                {8, 7},
            },
        .score = 2,
        .name = "#3-1",
        .size = 6,
    },
    {
        .moves =
            {
                {7, 7},
                {6, 8},
                {6, 7},
                {7, 6},
                {8, 7},
                {5, 7},
            },
        .score = 2,
        .name = "#3-2",
        .size = 6,
    },
    {
        .moves =
            {
                {7, 7},
                {6, 8},
                {6, 6},
                {8, 8},
                {7, 8},
                {7, 6},
                {5, 7},
                {8, 7},
            },
        .score = 4,
        .name = "#4-0",
        .size = 6,
    },
    {
        .moves =
            {
                {7, 7},
                {6, 8},
                {6, 6},
                {8, 8},
                {5, 8},
                {5, 6},
            },
        .score = 4,
        .name = "#4-1",
        .size = 6,
    },
    {
        .moves =
            {
                {7, 7},
                {6, 8},
                {6, 6},
                {8, 8},
                {9, 8},
                {5, 7},
            },
        .score = 4,
        .name = "#4-2",
        .size = 6,
    },
};

static const int black_opening_count = sizeof(black_openings) / sizeof(opening_t);
static const int white_opening_count = sizeof(white_openings) / sizeof(opening_t);

void test_opening() {
    for (int i = 0; i < black_opening_count; i++) {
        game_t game = new_game(1000);
        for (int j = 0; j < black_openings[i].size; j++) {
            add_step(&game, black_openings[i].moves[j]);
        }
        print_game(game);
        double start_time = record_time();
        vector_t vct_sequence = complex_vct(false, game.board, 1, 1000, 3);
        if (vct_sequence.size) {
            log_l("found VCT in %.2lfms", get_time(start_time));
            print_points(vct_sequence, PROMPT_NOTE, " -> ");
        }
        vector_free(vct_sequence);
    }
}

static point_t rotate_point(point_t p, int rotation) {
    point_t center = {7, 7};
    point_t rotated = p;

    rotated.x -= center.x;
    rotated.y -= center.y;

    for (int i = 0; i < rotation; i++) {
        int temp = rotated.x;
        rotated.x = -rotated.y;
        rotated.y = temp;
    }

    rotated.x += center.x;
    rotated.y += center.y;

    return rotated;
}

static point_t mirror_point(point_t p, bool mirror) {
    if (!mirror) return p;
    point_t center = {7, 7};
    return (point_t){center.x * 2 - p.x, p.y};
}

static bool check_opening_match(board_t board, const opening_t* opening, int depth, int rotation,
                                bool mirror, point_t* out_move) {
    if (depth >= opening->size) return false;
    board_t expected_board = {0};
    for (int j = 0; j < depth; j++) {
        point_t p = opening->moves[j];
        p = rotate_point(p, rotation);
        p = mirror_point(p, mirror);
        int expect_id = (j % 2 == 0) ? 1 : 2;
        expected_board[p.x][p.y] = expect_id;
    }
    if (!is_equal(board, expected_board)) return false;
    point_t next_move = opening->moves[depth];
    next_move = rotate_point(next_move, rotation);
    next_move = mirror_point(next_move, mirror);
    *out_move = next_move;
    return true;
}

/// @brief find a move from opening book
/// @param board current board state
/// @param depth current search depth
/// @param player_id current player's ID
/// @return best move
static point_t find_opening_move(board_t board, int depth, int player_id) {
    point_t best_move = {-1, -1};
    int best_score = -1000;
    const opening_t* openings = player_id == 1 ? black_openings : white_openings;
    int opening_count = player_id == 1 ? black_opening_count : white_opening_count;

    for (int i = 0; i < opening_count; i++) {
        for (int rotation = 0; rotation < 4; rotation++) {
            for (int mirror = 0; mirror <= 1; mirror++) {
                point_t move;
                if (check_opening_match(board, &openings[i], depth, rotation, mirror, &move)) {
                    if (openings[i].score > best_score) {
                        best_score = openings[i].score;
                        best_move = move;
                        log_l("find opening: %s, score: %d", openings[i].name, best_score);
                    }
                }
            }
        }
    }

    return best_move;
}

/// @brief find a trivial move from {game.board}
/// @param game current game state
/// @param time_limit time limit for search
/// @param use_opening whether to use opening book
/// @param use_vct whether to use VCT
/// @return best move
point_t trivial_move(game_t game, double time_limit, bool use_opening, bool use_vct) {
    const int self_id = game.cur_id;
    const int oppo_id = 3 - self_id;
    vector_t self_5 = vector_new(threat_t, NULL);
    vector_t self_4 = vector_new(threat_t, NULL);
    scan_threats(game.board, self_id, self_id,
                 (threat_storage_t){[PAT_WIN] = &self_5, [PAT_A4] = &self_4});
    vector_t oppo_5 = vector_new(threat_t, NULL);
    scan_threats(game.board, oppo_id, oppo_id, (threat_storage_t){[PAT_WIN] = &oppo_5});
    point_t pos = {-1, -1};
    bool is_attack;
    if (self_5.size) {
        threat_t attack = vector_get(threat_t, self_5, 0);
        pos = attack.pos, is_attack = true;
    }
    if (!in_board(pos) && oppo_5.size) {
        for_each(threat_t, oppo_5, defend) {
            if (!is_forbidden(game.board, defend.pos, self_id, 3)) {
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

    if (use_opening) {
        point_t opening_pos = find_opening_move(game.board, game.count, self_id);
        if (in_board(opening_pos)) {
            log_l("found opening move: %c%d", READABLE_POS(opening_pos));
            return opening_pos;
        }
    }

    if (use_vct) {
        double start_time = record_time();
        vector_t vct_sequence = complex_vct(false, game.board, self_id, time_limit / 2, 2);
        double duration = get_time(start_time);
        if (!vct_sequence.size) {
            vector_free(vct_sequence);
            vct_sequence = complex_vct(false, game.board, self_id, time_limit - duration, 3);
            // if (vct_sequence.size) {
            //     log_l("find in 3 depth but not in 2 depth!");
            //     prompt_pause();
            // }
        }
        if (vct_sequence.size) {
            pos = vector_get(point_t, vct_sequence, 0);
            log_l("found VCT in %.2lfms", get_time(start_time));
            print_points(vct_sequence, PROMPT_NOTE, " -> ");
            // sleep(1);
        } else {
            log_l("VCT not found, %.2lfms", get_time(start_time));
        }
        vector_free(vct_sequence);
    }
    return pos;
}