#include "board.h"
#include "eval.h"
#include "server.h"
#include "game.h"
#include "pattern.h"
#include "players.h"
#include "util.h"
#include "vct.h"
#include "vector.h"

#include <stdlib.h>
#include <string.h>

void test_eval() {
#include "boards.txt"
    vector_t result_5 = vector_new(threat_t, NULL), result_4 = vector_new(threat_t, NULL),
             result_3 = vector_new(threat_t, NULL);
    for (int i = 0; i < 1; i++) {
        board_t board;
        point_t pos;
        pos = parse_board(board, tests[i].str);
        result_3.size = result_4.size = result_5.size = 0;
        threat_storage_t storage = {[PAT_A3] = &result_3,
                                    [PAT_A4] = &result_4,
                                    [PAT_D4] = &result_4,
                                    [PAT_WIN] = &result_5};

        double tim = record_time();
        for (int t = 0; t < 1; t++) {
            scan_threats(board, 1, 1, storage);
        }
        tim = get_time(tim);
        log_l("time: %.3lfms", tim * 1.0);
        log_l("i = %d", i);
        print_emph(board, pos);
        fboard_t mark = {0};
        for_each(threat_t, result_3, threat) {
            point_t point = threat.pos;
            log_l("got [%s] %d, %d, dir = {%d, %d}", pattern_typename[threat.pattern], point.x,
                point.y, threat.dir.x, threat.dir.y);
            mark[point.x][point.y] = 0.05;
        }
        print_prob(board, mark);
        memset(mark, 0, sizeof(mark));
        print_prob(board, mark);
        for_each(threat_t, result_4, threat) {
            point_t point = threat.pos;
            log_l("got [%s] %d, %d, dir = {%d, %d}", pattern_typename[threat.pattern], point.x,
                point.y, threat.dir.x, threat.dir.y);
            mark[point.x][point.y] = 0.2;
        }
        print_prob(board, mark);
        memset(mark, 0, sizeof(mark));
        for_each(threat_t, result_5, threat) {
            point_t point = threat.pos;
            log_l("got [%s] %d, %d, dir = {%d, %d}", pattern_typename[threat.pattern], point.x,
                point.y, threat.dir.x, threat.dir.y);
            mark[point.x][point.y] = 0.4;
        }
        print_prob(board, mark);
    }
    vector_free(result_3), vector_free(result_4), vector_free(result_5);
}

void test_upd() {
#include "boards.txt"
    for (int i = TESTS - 1; i < TESTS; i++) {
        board_t board;
        point_t pos;
        pos = parse_board(board, tests[i].str);
        print_emph(board, pos);
        vector_t array = vector_new(point_t, NULL);
        board[pos.x][pos.y] = 2;
        for_all_dir(d, dx, dy) {
            vector_t tmp = find_relative_points(ATTACK, board, pos, dx, dy, 2, false);
            print_emph_mutiple(board, tmp);
            vector_cat(array, tmp);
            vector_free(tmp);
        }
        print_emph_mutiple(board, array);
    }
}

void test_threat() {
    // clang-format off
    game_t game = restore_game(2000,18,(point_t[]){{7,7},{6,6},{5,7},{4,7},{6,8},{4,6},{5,9},{4,10},{7,8},{7,6},{8,6},{9,5},{8,7},{4,9},{4,8},{5,8},{6,7},{9,7}});
    print_game(game);
    // clang-format on
    // vector_t a3 = vector_new(threat_t, NULL);
    // scan_threats(game.board, 1, 1, (threat_storage_t){[PAT_A3] = &a3});
    vector_t a3 = scan_threats_by_threshold(game.board, 1, PAT_A3);
    for_each(threat_t, a3, a) { print_emph(game.board, a.pos); }
    double t = record_time();
    for (int i = 0; i < 100000; i++) {
        vector_t a3 = scan_threats_by_threshold(game.board, 1, PAT_A3);
    }
    log_l("time: %.2lf ms", get_time(t));
}

void test_threat_tree() {
    // clang-format off
    game_t game = 
    restore_game(15000,29,(point_t[]){{7,7},{7,8},{6,6},{8,8},{6,8},{8,6},{6,7},{6,5},{8,7},{5,7},{5,9},{4,10},{6,10},{6,9},{5,8},{9,7},{3,8},{4,8},{4,9},{3,10},{5,10},{6,11},{7,6},{8,5},{7,5},{7,4},{2,7},{1,6},{5,6}});
    /*
    [G10] (H11) [I10] (J10) [I11] (I12) [E10] (F10) [E9] (E7) 
    '*/
    game_t game2 = 
    restore_game(10000,18,(point_t[]){{7,7},{6,7},{6,6},{5,5},{7,5},{8,4},{7,6},{7,8},{5,6},{8,6},{4,7},{3,8},{4,6},{3,6},{4,8},{4,9},{8,7},{9,8}});
    game_t game3 = 
    restore_game(10000,20,(point_t[]){{7,7},{6,7},{6,6},{5,5},{7,5},{8,4},{7,6},{7,8},{5,6},{8,6},{4,7},{3,8},{4,6},{3,6},{4,8},{4,9},{8,7},{9,8},{7,4},{7,3}});
    // char buffer[1024];
    // board_serialize(game.board, buffer);
    // printf("%s", buffer);
    game2 = game3;
    print_game(game2);
    double st = record_time();
    vector_t vct2 = vct(false, game2.board, game2.cur_id, 5000);
    double du = get_time(st);
    print_points(vct2, PROMPT_LOG, " -> ");
    vector_free(vct2);
    log_l("time: %.2lfms", du);
    start_game(preset_players[MANUAL], preset_players[MINIMAX_ADV], 1, 10000, &game2, NULL);
    // clang-format on

    // #include "boards.txt"
    // for (int i = 0; i < 1; i++) {
    //     board_t board;
    //     point_t pos;
    //     pos = parse_board(board, tests[i].str);
    //     print_emph(board, pos);
    //     vector_t threat_info = scan_threats_info(board, 1, true);
    //     for_each(threat_info_t, threat_info, info) {
    //         log_l("=========================================");
    //         log_l("type: %s", pattern_typename[info.type]);
    //         log_l("pos: (%d, %d)", info.action.x, info.action.y);
    //         print_emph(board, info.action);
    //         log_l("defends:");
    //         print_emph_mutiple(board, info.defenses);
    //     }
    // }
}

void test_threat_seq() {
#include "board_vct.txt"
    for (int T = 0; T < 10; T++) {
        for (int i = 0; i < VCT_TESTS; i++) {
            // for (int i = 0; i < 1; i++) {
            int id = tests[i].id;
            // prompt_scanf("%d%d", &i, &id);
            board_t board;
            parse_board(board, tests[i].str);
            // for (int i = 0; i < BOARD_SIZE; i++) {
            //     for (int j = 0; j < BOARD_SIZE; j++) {
            //         if (board[i][j]) board[i][j] = 3 - board[i][j];
            //     }
            // }
            // id = 3 - id;
            // vector_t forest = get_threat_forest(board, 1, true);
            // for_each_ptr(threat_tree_node_t*, forest, proot) {
            //     threat_tree_node_t* root = *proot;
            //     print_threat_tree(root);
            // }
            print(board);
            double start_time = record_time();
            vector_t seq = vct(false, board, id, 5000);
            print_points(seq, PROMPT_LOG, " -> ");
            log_l("time: %.3lfms", get_time(start_time));
            int get_vct = seq.size != 0;
            vector_free(seq);
            if (get_vct != tests[i].have_vct) {
                log_e("test failed, expected %d, got %d", tests[i].have_vct, get_vct);
                exit(EXIT_FAILURE);
            }
            // point_t p = vct(board, 1, false);
            // if (in_board(p)) {
            //     log_l("result: %c%d", READABLE_POS(p));
            // } else {
            //     log_l("no result");
            // }
        }
    }
}

/*
restore_game(2000,14,(point_t[]){{7,7},{4,8},{6,6},{8,8},{7,5},{7,9},{7,6},{7,8},{5,6},{8,6},{7,4},{7,3},{6,5},{4,7}});
wrong vct: D7 -> E6 -> G4 -> F5
*/
