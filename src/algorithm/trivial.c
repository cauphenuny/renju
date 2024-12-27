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

static const opening_t openings[] = {
    // 开局1: 黑天元 - 白天元防 - 黑花月 - 白反花月 - 黑三手 - 白防守 - 黑进攻 - 白防守
    {.moves =
         {
             {7, 7},  // 天元
             {8, 8},  // 天元防
             {7, 8},  // 花月
             {6, 7},  // 反花月
             {8, 7},  // 黑三手
             {6, 8},  // 白防守
             {9, 7},  // 黑进攻
             {5, 7}   // 白防守
         },
     .size = 8,
     .score = 3,  // 黑棋较优
     .name = "天元花月进攻型"},

    // 开局2: 黑天元 - 白一路防 - 黑花月 - 白反花月 - 黑连击 - 白防守 - 黑进攻 - 白防守
    {.moves =
         {
             {7, 7},  // 天元
             {7, 8},  // 一路防
             {7, 6},  // 花月
             {8, 7},  // 反花月
             {6, 7},  // 黑连击
             {8, 6},  // 白防守
             {6, 8},  // 黑进攻
             {8, 8}   // 白防守
         },
     .size = 8,
     .score = -2,  // 白棋较优
     .name = "一路花月防守型"},

    // 开局3: 黑天元 - 白二路防 - 黑花月 - 白反花月 - 黑斜进 - 白防守 - 黑压制 - 白缓冲
    {.moves =
         {
             {7, 7},  // 天元
             {6, 7},  // 二路防
             {8, 7},  // 花月
             {7, 8},  // 反花月
             {8, 6},  // 黑斜进
             {6, 8},  // 白防守
             {9, 6},  // 黑压制
             {7, 6}   // 白缓冲
         },
     .size = 8,
     .score = 1,  // 黑棋略优
     .name = "二路花月斜进型"},

    // 开局4: 黑天元 - 白三路防 - 黑花月 - 白反花月 - 黑包围 - 白突破 - 黑补防 - 白反击
    {.moves =
         {
             {7, 7},  // 天元
             {5, 7},  // 三路防
             {8, 7},  // 花月
             {7, 8},  // 反花月
             {6, 6},  // 黑包围
             {7, 6},  // 白突破
             {6, 8},  // 黑补防
             {8, 6}   // 白反击
         },
     .size = 8,
     .score = -1,  // 白棋略优
     .name = "三路花月包围型"},

    // 开局5: 黑天元 - 白斜防 - 黑花月 - 白反花月 - 黑压制 - 白防守 - 黑进攻 - 白防守
    {.moves =
         {
             {7, 7},  // 天元
             {8, 6},  // 斜防
             {6, 8},  // 花月
             {7, 8},  // 反花月
             {8, 8},  // 黑压制
             {6, 7},  // 白防守
             {9, 8},  // 黑进攻
             {5, 8}   // 白防守
         },
     .size = 8,
     .score = 2,  // 黑棋优势
     .name = "斜防花月压制型"},

    // 开局6: 黑天元 - 白远角防 - 黑花月 - 白反花月 - 黑连击 - 白防守 - 黑进攻 - 白防守
    {.moves =
         {
             {7, 7},  // 天元
             {9, 9},  // 远角防
             {8, 8},  // 花月
             {7, 8},  // 反花月
             {6, 8},  // 黑连击
             {8, 7},  // 白防守
             {6, 9},  // 黑进攻
             {8, 6}   // 白防守
         },
     .size = 8,
     .score = 0,  // 均势
     .name = "远角花月连击型"},

    // 新增开局7: 黑天元 - 白近角防 - 黑花月 - 白反花月 - 黑包围 - 白防守 - 黑进攻 - 白防守
    {.moves =
         {
             {7, 7},  // 天元
             {8, 8},  // 近角防
             {6, 8},  // 花月
             {7, 8},  // 反花月
             {8, 7},  // 黑包围
             {6, 7},  // 白防守
             {9, 7},  // 黑进攻
             {5, 8}   // 白防守
         },
     .size = 8,
     .score = 1,  // 黑棋略优
     .name = "近角花月包围型"},

    // 新增开局8: 黑天元 - 白中路防 - 黑花月 - 白反花月 - 黑连击 - 白防守 - 黑压制 - 白防守
    {.moves =
         {
             {7, 7},  // 天元
             {7, 6},  // 中路防
             {8, 7},  // 花月
             {7, 8},  // 反花月
             {8, 8},  // 黑连击
             {6, 7},  // 白防守
             {9, 8},  // 黑压制
             {6, 8}   // 白防守
         },
     .size = 8,
     .score = 2,  // 黑棋优势
     .name = "中路花月连击型"}};

static const int opening_count = sizeof(openings) / sizeof(opening_t);

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
    for (int j = 0; j < depth; j++) {
        point_t p = opening->moves[j];
        p = rotate_point(p, rotation);
        p = mirror_point(p, mirror);
        if (!board[p.x][p.y]) {
            return false;
        }
    }

    if (depth < opening->size) {
        point_t next_move = opening->moves[depth];
        next_move = rotate_point(next_move, rotation);
        next_move = mirror_point(next_move, mirror);
        *out_move = next_move;
        return true;
    }
    return false;
}

/// @brief find a move from opening book
/// @param board current board state
/// @param depth current search depth
/// @param player_id current player's ID
/// @return best move
static point_t find_opening_move(board_t board, int depth, int player_id) {
    point_t best_move = {-1, -1};
    int best_score = player_id == 1 ? -1000 : 1000;

    for (int i = 0; i < opening_count; i++) {
        if (openings[i].score < 0 && player_id == 1) continue;
        if (openings[i].score > 0 && player_id == 2) continue;
        for (int rotation = 0; rotation < 4; rotation++) {
            for (int mirror = 0; mirror <= 1; mirror++) {
                point_t move;
                if (check_opening_match(board, &openings[i], depth, rotation, mirror, &move)) {
                    if (player_id == 1) {
                        if (openings[i].score > best_score) {
                            best_score = openings[i].score;
                            best_move = move;
                            log_l("find opening: %s, score: %d", openings[i].name, best_score);
                        }
                    } else {
                        if (openings[i].score < best_score) {
                            best_score = openings[i].score;
                            best_move = move;
                            log_l("find opening: %s, score: %d", openings[i].name, best_score);
                        }
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
        if (!vct_sequence.size) {
            vector_free(vct_sequence);
            vct_sequence = complex_vct(false, game.board, self_id, time_limit / 2, 3);
            if (vct_sequence.size) {
                log_l("find in 3 depth but not in 2 depth!");
                prompt_pause();
            }
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